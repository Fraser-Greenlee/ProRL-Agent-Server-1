# Plan: ECHO env-CE + targeted self-distillation (B3) on Polar + Slime

This document is a self-contained implementation plan. The implementer should
not need additional research outside the three repositories named below.

## 0 · Audience and scope

You are implementing **two complementary auxiliary losses** layered on top of an
existing GRPO trainer that runs on Polar rollouts:

1. **ECHO** — cross-entropy on environment tokens (free per-token supervision
   on tool/user observations). Single forward pass; one extra mask.
2. **Targeted self-distillation (B3)** — KL/JSD on the **student's actually
   sampled tokens at one or two mistaken turns**, with the teacher being the
   same policy conditioned on the student's prefix plus an inserted hint.
   Single extra teacher forward; sparse mask.

Both losses are added on top of the existing GRPO/PPO loss inside one trainer
worker. ECHO has no extra rollouts; B3 has at most one cheap hint-generation
rollout per low-reward sample.

The two losses are independent and can be enabled/disabled per run with
coefficients. Land them in this order: schema changes, ECHO, B3.

## 1 · Repositories you must read

You only need three repos. Read these files end-to-end before writing code.

### 1.1 Polar (this repo, `/Users/fgreenlee/Projects/ProRL-Agent-Server`)

The rollout server. You will modify it minimally — only schema + builders.

Required reading:
- `src/polar/trajectory/models.py` — `Trace`, `Trajectory`, `CompletionRecord`,
  `CompletionSession`. The schema you'll extend.
- `src/polar/trajectory/builder/per_request.py` — simple builder, one trace per
  completion record. Fine starting point if you don't have multi-turn chains.
- `src/polar/trajectory/builder/prefix_merging.py` — the production builder.
  Stitches multi-turn chains into one trace via canonical-prefix matching.
  **Read the whole file**, including the docstring: it explains why we mix
  raw `response_ids` (sampled) with canonical interstitial tokens (server
  tokenization) and never decode → re-encode. You must follow the same
  discipline when you splice the hint.
- `src/polar/trajectory/builder/record_utils.py` — `build_trace_from_completion`.
  Where `prompt_ids` / `response_ids` / `response_logprobs` come from.
- `src/polar/trajectory/registry.py` — how builders are registered by name.
  You'll register one new builder.
- `src/polar/agent/README.md` — agents/harnesses/proxy. Skim only.
- `src/polar/rollout/README.md` — task → sessions → callback. Skim only.
- `src/slime_bridge/adapter.py` and `src/slime_bridge/_messages.py` — how a
  Polar `SessionResult` becomes Slime `Sample`s today. You'll extend this.
- `src/slime_bridge/rollout.py` — async rollout worker, the place that submits
  Polar tasks and receives terminal results. You'll add the SDPO submission
  pass here.

You do **not** need to touch `polar.gateway`, `polar.runtime`, or
`polar.platform`. The proxy already captures token ids and logprobs.

### 1.2 ECHO (`/Users/fgreenlee/Projects/echo-rl`)

The reference implementation of env-CE. We are *re-implementing* its idea on
top of Slime, not copying its code (it lives on top of SkyRL). Read for
mechanism, not for verbatim translation.

Required reading:
- `README.md` — full design intent.
- `echo_rl/world_modeling/loss.py` — the entire CE-on-mask loss, normalization
  variants, and metrics. We will mirror this almost verbatim.
- `echo_rl/world_modeling/fsdp_worker.py` — `get_aux_policy_loss_context` and
  `compute_aux_policy_loss`. The hook shape we need on the Slime side.
- `echo_rl/world_modeling/config.py` — coefficient schedule fields.
- `echo_rl/terminal_agent/interaction.py` — the dataclass that carries parallel
  masks (`completion_masks`, `completion_observation_masks`,
  `completion_warning_masks`, `completion_env_output_masks`). Same idea we
  will add to Polar's `Trace`.
- `echo_rl/terminal_agent/terminal_agent_generator.py` lines 480–560 — how
  ECHO computes the message-content span (start/end of body, excluding
  chat-template wrappers) by re-tokenizing the same messages list with and
  without the new turn. **This is the technique you must use** when you add
  env tokens to `response_ids`.
- `patches/skyrl_minimal_hooks.patch` — read fully. It's ~80 lines and is the
  template for what we patch into Slime: `extras` on `Experience`, a
  `zero_pad_keys` mechanism for padding aux tensors with zeros, and the
  `get_aux_policy_loss_context` / `compute_aux_policy_loss` hooks on the
  policy worker. Slime needs an analogous patch.

### 1.3 SDPO (`/Users/fgreenlee/Projects/SDPO`)

Reference implementation of self-distillation with reprompted teacher. We are
copying its loss math verbatim and replacing its prompt-level mask with a
**token-level turn-k mask**.

Required reading:
- `README.md` — mechanism summary and config glossary.
- `verl/trainer/ppo/core_algos.py` — `compute_self_distillation_loss`
  (around line 1085). **This entire function is portable**: top-k bucket with
  tail, full-logit KL, alpha-interpolated JSD, IS clipping. Lift it.
- `verl/workers/actor/dp_actor.py` lines 680–850 — how SDPO runs the teacher
  forward inside the same actor step:
  - `teacher_inputs = {responses, input_ids: teacher_input_ids,
    attention_mask: teacher_attention_mask, position_ids: teacher_position_ids}`
  - `teacher_model = self.teacher_module or self.actor_module`
  - `with torch.no_grad(): teacher_outputs = self._forward_micro_batch(...)`
  - then `compute_self_distillation_loss(...)`.
  Mirror this structure on the Slime worker.
- `verl/trainer/ppo/ray_trainer.py` lines 640–800 — `_maybe_build_self_distillation_batch`,
  `_collect_solutions_by_uid`, `_get_solution`. SDPO assembles the teacher
  prompt at the **trainer** layer (post-rollout, pre-update). We will do the
  same, except our teacher prompt is built in the **bridge**, not the trainer,
  because we already have full token-level Polar trajectories.
- `verl/trainer/config/actor/actor.yaml` — every SDPO config name we should
  preserve to avoid bikeshedding (e.g. `alpha`, `is_clip`, `distillation_topk`,
  `teacher_update_rate`, `teacher_regularization: ema`).

You do **not** need to touch SDPO. You are reading it.

## 2 · The mental model in one diagram

For one trajectory with two assistant turns (the agent ran two LLM calls; one
env response between them) and a wrong outcome:

```
position →   prompt | resp_1 tokens | env_1 tokens | resp_2 tokens
                     ←──── trainable assistant tokens ────→
loss_mask    0...0   1 1 1 1 1 1 1   0 0 0 0 0 0 0  1 1 1 1 1 1 1
env_mask     0...0   0 0 0 0 0 0 0   0 1 1 1 1 1 0  0 0 0 0 0 0 0
                                       ↑ env body only, chat-template wrappers stay 0
sd_mask      0...0   0 0 0 0 0 0 0   0 0 0 0 0 0 0  1 1 1 1 1 1 1
                                                    ↑ turn flagged as "mistaken"
```

Three masks, one length each (== `len(response_ids)`). The same forward over
`(prompt + response)` produces:

- Standard policy gradient on positions where `loss_mask == 1`.
- ECHO CE on positions where `env_mask == 1` (cheap; same forward).
- SDPO targeted KL on positions where `sd_mask == 1`. Computed by running a
  *second* forward (the teacher) over a different but token-aligned sequence
  whose tokens at the sd-mask positions are *the same* as the student's. KL
  between student logits and teacher logits at those positions only.

This is the entire architectural payload. Everything else is plumbing.

## 3 · End-to-end data flow

```
  ┌── Polar rollouts ─────────────────────────────────────────────┐
  │ 1. Trainer (Slime) submits a TaskRequest per group.          │
  │ 2. Each session runs in a sandbox, agent calls flow through  │
  │    the gateway proxy, completions are captured with token    │
  │    ids + logprobs.                                           │
  │ 3. POST-RUN: builder turns CompletionSession → Trajectory.   │
  │    *NEW: this builder also emits env_loss_mask and a         │
  │    per-turn offset table inside Trace.metadata.*             │
  │ 4. Evaluator scores the trajectory; reward attached to each  │
  │    Trace.                                                    │
  │ 5. Polar callback delivers TaskResult to Slime bridge.       │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌── Slime bridge ───────────────────────────────────────────────┐
  │ 6. *NEW: hint stage* — for low-reward sessions, derive a     │
  │    (turn_index, hint_text) pair. Two interchangeable hint    │
  │    sources: deterministic (test stack trace, compile error)  │
  │    or LLM (one extra Polar task with a critic harness).      │
  │ 7. *NEW: teacher prefix builder* — splice hint into the      │
  │    student's prefix at turn k, producing teacher_input_ids,  │
  │    teacher_attention_mask, sd_target_mask, plus offsets.     │
  │ 8. adapter.session_result_to_samples emits a Slime Sample    │
  │    per Trace with extras: env_loss_mask, sd_target_mask,     │
  │    teacher_input_ids, teacher_attention_mask, alignment      │
  │    offsets, world_full_observation_count.                    │
  │ 9. post_process_rewards (existing) does GRPO normalization   │
  │    at trajectory granularity. Unchanged.                     │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌── Slime trainer (with the upstream patch we add) ─────────────┐
  │ 10. *NEW: aux-loss hook* — student forward (existing); if    │
  │     env_loss_mask present, add ECHO loss; if teacher tensors │
  │     present, run teacher forward (no_grad if EMA, with grad  │
  │     stop on teacher in any case) and add SDPO loss masked by │
  │     sd_target_mask. Total = pg_loss + λ_env·echo + λ_sd·sdpo.│
  │ 11. EMA-update teacher weights every step (mirror SDPO).     │
  └───────────────────────────────────────────────────────────────┘
```

## 4 · Schema changes (Polar)

### 4.1 `Trace` — `src/polar/trajectory/models.py`

Add three fields to `Trace`. Same length contract as `loss_mask` (validated
against `len(response_ids)`). All optional for backward compatibility.

```python
class Trace(BaseModel):
    # ... existing fields ...
    env_loss_mask: list[int] = Field(default_factory=list)   # ECHO env CE mask
    turn_offsets: list[tuple[int, int]] = Field(default_factory=list)
        # list of (start, end) into response_ids, one entry per assistant turn.
        # Used by the hint-splicing step in the bridge to locate turn-k tokens.
        # Empty when builder doesn't track turns (per_request).
    full_observation_count: int = 0
        # Total env body tokens (used by ECHO's full_observation_tokens
        # normalization). Does not include chat-template wrappers.
```

Validators (mirror the existing `loss_mask` validator):

```python
@field_validator("env_loss_mask")
@classmethod
def _validate_env_mask_values(cls, value: list[int]) -> list[int]:
    out = []
    for v in value:
        if int(v) not in (0, 1):
            raise ValueError("env_loss_mask values must be 0 or 1")
        out.append(int(v))
    return out

@model_validator(mode="after")
def _validate_response_lengths(self) -> "Trace":
    # ... keep existing checks ...
    if self.env_loss_mask and len(self.env_loss_mask) != len(self.response_ids):
        raise ValueError("env_loss_mask length must match response_ids length")
    return self
```

Do not change `loss_mask` semantics. ECHO's mask is **separate** from
`loss_mask` and `loss_mask` stays purely "trainable assistant tokens".

### 4.2 No other Polar schema changes

Do **not** add SDPO fields to `Trace`. The teacher tensors and `sd_target_mask`
are constructed in the bridge from the student trace; storing them on `Trace`
would couple Polar to a specific trainer.

## 5 · Builder changes (Polar)

You need one new builder. Name it `prefix_merging_with_env`. Register it in
`src/polar/trajectory/registry.py` alongside `prefix_merging`.

Place: `src/polar/trajectory/builder/prefix_merging_with_env.py`. Subclass
`PrefixMergingBuilder`. Override `_finalize_chain` to do everything the parent
does **plus**:

1. **Splice env tokens into the trainable response.** Today, the parent puts
   interstitial tokens (env messages + chat glue) into `stream_ids` with
   `loss_mask = 0`. Keep that. The set of positions that fall into
   "interstitial" is exactly the set of candidate env-CE positions.

2. **Compute the env body span within each interstitial.** This is the
   non-trivial part. The interstitial as currently sliced
   (`_slice_interstitial`) starts at the first `<|im_end|>` of the previous
   assistant turn and runs to the next generation prompt. It contains:
   - the prev-turn `<|im_end|>` (or not, depending on natural-stop branch),
   - one or more chat-template message wrappers like `<|im_start|>tool\n…<|im_end|>`,
   - generation-prompt glue like `<|im_start|>assistant\n`.

   You need to mark **only the body bytes** between `<|im_start|>tool\n` and
   `<|im_end|>` as env-mask = 1. Everything else stays 0.

   Use ECHO's technique (terminal_agent_generator.py:519-556). For each
   tool/user message you know was inserted between turn k and turn k+1
   (you can read these from `Ci_trace.prompt_messages[msg_acc:]` — the
   parent builder already iterates these), tokenize *the same messages list*
   three times via the chat template:
   - `before` = messages so far, no new message appended.
   - `after_msg` = messages + this one message, **without** generation prompt.
   - `after_full` = messages + this one message, **with** generation prompt.

   `body_token_count` = `len(after_msg) - len(before) - len(role_wrapper_prefix) - len(role_wrapper_suffix)`.

   The cleaner and BPE-safer approach actually mirrors the chain-prefix logic:
   the `after_msg` token sequence minus `before` = the wrapper + body + closing
   tokens for that one message. You then locate the body span inside that
   delta by re-encoding `text` (the message content) under the chat template's
   role marker. ECHO does this; copy the structure of `_get_observation_content_spans`.

   Result: for every position you appended into `stream_ids` that came from
   tool/user message bodies, set `env_loss_mask` to 1. Everything else is 0.

3. **Fill `turn_offsets`**. After all turns are spliced, walk the response
   stream and record `(start, end)` for each assistant turn (you know each
   assistant turn's length from `Ci_trace.response_ids` length). Store on the
   final `Trace`.

4. **Fill `full_observation_count`**. Sum the body lengths over all messages
   touched. Used as ECHO's normalizer.

The parent builder's correctness rules apply. Do not decode→re-encode any
sampled response tokens. Do not assume tokenizer-specific role markers; use
the tokenizer's `apply_chat_template` everywhere. The Builder receives the
tokenizer via dependency injection — see how `prefix_merging` is constructed
in the registry; if it needs a tokenizer it can take one through `__init__`
(the gateway has access via the inference engine).

If you cannot reach the tokenizer cleanly inside the builder, fall back: emit
empty `env_loss_mask` for that interstitial and log a warning. **Do not
guess the body span heuristically** — wrong spans here silently corrupt the
ECHO loss.

### Tests

Add `tests/trajectory/test_prefix_merging_with_env.py`:
- A captured two-turn session fixture (use ChatML / Qwen tokenizer) with one
  tool response in between. Assert:
  - `len(env_loss_mask) == len(response_ids)`.
  - `sum(env_loss_mask) > 0`.
  - For every position with `env_loss_mask == 1`, `loss_mask == 0`.
  - For every position with `loss_mask == 1`, `env_loss_mask == 0`.
  - `turn_offsets` has 2 entries; assistant tokens at those slices are exactly
    the original `response_ids` of each completion.
  - `full_observation_count` equals the actual body token count.
- A compaction case: when the next prompt is *not* a token-prefix of the
  previous, a new chain starts (parent behavior). Both chains' env masks should
  cover their own interstitials independently.

## 6 · Bridge changes (slime_bridge)

### 6.1 Carry the new fields onto Slime samples

`src/slime_bridge/adapter.py`, `session_result_to_samples`:

- Read `trace.env_loss_mask` and attach to the Slime `Sample` via whatever
  extras hook Slime exposes. If the existing Sample type does not have an
  `extras` dict, add one as part of the Slime patch (see §8).
- Attach `trace.turn_offsets` and `trace.full_observation_count` likewise.

Do **not** drop traces that have empty `env_loss_mask` — they still train via
GRPO. ECHO loss simply contributes zero on them.

### 6.2 Hint generation stage

New module: `src/slime_bridge/sdpo.py`.

Public surface:

```python
@dataclass(slots=True, frozen=True)
class HintRecord:
    turn_index: int            # 0-based index into trace.turn_offsets
    hint_text: str             # string to insert as a chat message
    hint_role: str = "user"    # role for the inserted message
    source: str = "deterministic"  # "deterministic" | "llm" | other tag for metrics

async def maybe_produce_hint(
    *,
    session_result: SessionResult,
    trace: Trace,
    rollout_url: str,            # for the optional LLM hint
    config: SdpoBridgeConfig,
) -> HintRecord | None:
    ...
```

Hint sources, both behind one config flag:

1. **Deterministic** (default; cheap; ship first).
   - If reward >= success_threshold: return None.
   - Find the failing test / first compile error / first nonzero exit code in
     the evaluator metadata or `eval_artifacts/`. Polar already exposes this
     via `SessionResult.metadata` and the on-disk
     `task_<id>/sessions/<sid>/eval/...` files. Read whichever the existing
     `slime_bridge` rollout already has access to.
   - Identify the assistant turn whose tool call produced the failing state.
     Easiest correct rule: the **last assistant turn before the failing eval
     command in trajectory order**. If you have per-tool-call exit codes
     in `Trace.response_messages` (you should — they're captured), pick the
     last assistant turn whose tool call had a non-zero exit code.
   - `hint_text` = a templated string with the captured stderr / failed
     assertion. Keep it short; truncate to N chars.

2. **LLM critic** (optional; behind `config.hint_source = "llm"`).
   - Submit a Polar `TaskRequest` with `agent.harness = "shell"` running a
     single curl/python call to the same inference server, prompt = "here is
     the trajectory and the reward; respond with JSON
     `{turn_index: int, hint_text: str}`".
   - Use Polar's existing rollout machinery. No new gateway code. The captured
     CompletionRecord on that auxiliary session is fine to discard at the
     bridge — you only consume the parsed JSON.
   - Validate the returned `turn_index` against `len(trace.turn_offsets)`; if
     out of range, fall back to deterministic.

Hint records are produced **per failed Trace**. Successful traces and traces
in groups with no failures get `None`.

### 6.3 Teacher prefix splicer

Same module. Public surface:

```python
@dataclass(slots=True)
class SdpoTensors:
    teacher_input_ids: list[int]
    teacher_attention_mask: list[int]
    sd_target_mask: list[int]            # length = len(teacher_input_ids), 1 on the
                                          # spliced student turn-k tokens, 0 elsewhere
    student_target_offset: int            # absolute offset of turn-k in student input_ids
    teacher_target_offset: int            # absolute offset of turn-k in teacher_input_ids
    target_length: int                    # number of tokens in turn-k

def build_teacher_inputs(
    *,
    trace: Trace,
    student_prompt_ids: list[int],
    student_response_ids: list[int],
    hint: HintRecord,
    tokenizer,
) -> SdpoTensors:
    ...
```

Implementation:

1. Look up turn-k slice from `trace.turn_offsets[hint.turn_index]` →
   `(s, e)`. Student turn-k tokens are
   `student_response_ids[s:e]`. Their absolute offset in the full student
   sequence is `len(student_prompt_ids) + s`.

2. Build the **teacher prefix** = student_response_ids[:s] preceded by
   student_prompt_ids, **then splice the hint message**. Use the same chat
   template canonicalization as the gateway: re-tokenize *only the inserted
   hint message* by calling `apply_chat_template` on
   `[..., {role: hint.hint_role, content: hint.hint_text}]` minus the prefix
   tokenization. Same trick as the env-mask body extractor.

   Concretely: `hint_tokens = tokenize_one_message(hint.hint_role,
   hint.hint_text)` where `tokenize_one_message` follows ECHO's three-encode
   delta technique to get the canonical token ids for that one message
   including its role wrappers and the generation-prompt that opens turn k.

3. Concatenate:
   ```
   teacher_input_ids = student_prompt_ids
                     + student_response_ids[:s]   # turns 0..k-1 with their env tokens
                     + hint_tokens                 # spliced message + generation prompt
                     + student_response_ids[s:e]   # student's actual turn-k tokens
   ```

   Note we deliberately stop at `e`, not at the end. We do **not** need turn
   k+1+ in the teacher sequence; it is unused (mask is zero past `e`). This
   keeps the teacher forward shorter.

4. Build `sd_target_mask` over the teacher sequence: zeros everywhere except
   ones at `[len(teacher_input_ids) - target_length : len(teacher_input_ids)]`.

5. Build `teacher_attention_mask` = ones over real tokens (no padding here;
   padding is the trainer's job).

6. Record offsets:
   ```
   student_target_offset = len(student_prompt_ids) + s
   teacher_target_offset = len(teacher_input_ids) - target_length
   target_length = e - s
   ```

   These offsets are how the trainer aligns the two forwards.

Verify by assertion: the last `target_length` tokens of `teacher_input_ids`
**must equal** `student_response_ids[s:e]`. If not, your splicing has an
off-by-one — abort the SDPO record for that trace and emit a metric.

### 6.4 Bridge wiring

In `src/slime_bridge/rollout.py`, after a `TaskResult` arrives and after
`session_result_to_samples` has produced the per-trace `Sample`s:

- For each session whose reward < `sdpo.success_threshold`:
  - Pick the trace to attach SDPO to. Default: the trace covering the failing
    turn (if you only have one trace per session, that's trivially it; if
    `prefix_merging_with_env` produced multiple traces because of compaction,
    the failing turn is in the last chain that has a tool call).
  - `hint = await maybe_produce_hint(...)`.
  - If `hint is None`: skip; sample's `sd_target_mask` stays empty. Otherwise:
  - `tensors = build_teacher_inputs(...)` and attach to the sample's extras.

- For successful sessions: attach nothing SDPO-related.

This entire stage is best-effort. Failures should log + drop SDPO on that
sample, never block GRPO.

### 6.5 Config

Add a `SdpoBridgeConfig` block with:

- `enabled: bool = False`
- `success_threshold: float = 1.0`
- `hint_source: Literal["deterministic", "llm"] = "deterministic"`
- `llm_critic_model: str | None = None`
- `llm_critic_harness: str = "shell"`
- `max_hint_chars: int = 2048`
- `max_teacher_seq_len: int = 16384` — drop the SDPO record if the spliced
  teacher exceeds this.

Wire it into `PolarSlimeConfig`. Make sure default off so existing runs are
unaffected.

## 7 · Loss math (drop into one file)

New file in your trainer-side code: `slime_bridge/aux_losses.py` (or wherever
your Slime extension package lives — see §8).

### 7.1 ECHO env-CE loss

Lift `compute_world_model_loss` from
`echo-rl/echo_rl/world_modeling/loss.py` verbatim. Adapt argument names to
your codebase if needed but do not change the math. Keep the three reduction
modes (`token_mean`, `sequence_mean`, `seq_mean_token_sum_norm`) and both
normalization options (`selected_tokens`, `full_observation_tokens`).

Coefficient schedule: lift `get_scheduled_coeff` verbatim from the same file.

Config object (mirror ECHO's `EchoAlgorithmConfig`):

- `world_model_coeff: float = 0.0`        # 0.0 disables ECHO
- `world_model_coeff_end: float = 0.0`
- `world_model_coeff_schedule: Literal["constant", "linear_decay", "step_decay"] = "constant"`
- `world_model_coeff_transition_step: int = 0`
- `world_loss_normalization: Literal["selected_tokens", "full_observation_tokens"] = "full_observation_tokens"`
- `loss_reduction: Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm"] = "token_mean"`
- `max_seq_len: int | None = None`        # required when seq_mean_token_sum_norm

### 7.2 SDPO targeted KL loss

Lift `compute_self_distillation_loss` verbatim from
`SDPO/verl/trainer/ppo/core_algos.py`. It is self-contained. Keep:

- `full_logit_distillation` + `distillation_topk` + `distillation_add_tail`
  (top-k bucket with tail probability for memory efficiency).
- `alpha` interpolation (0.0 = forward KL, 1.0 = reverse KL, 0.5 = JSD).
- `is_clip` for importance weighting against `old_log_probs`.
- `rollout_is_weights` if you compute them; pass `None` otherwise.

Replace its `response_mask` argument with our `sd_target_mask`. The function
multiplies `loss_mask = response_mask * self_distillation_mask.unsqueeze(1)`;
in our case `response_mask` should be your standard PPO response mask and
`self_distillation_mask` is a **per-token** mask (not per-sequence as in
SDPO). If your existing code expects per-sequence, pass an all-ones response
mask and put the per-token gating into `self_distillation_mask` (then the
unsqueeze still works because you'd unsqueeze on dim 1 over a tensor of
shape `[B, T]`, which is a no-op — read SDPO's code carefully here and fix
the dim if needed).

Config (mirror SDPO's `self_distillation`):

- `sdpo_coeff: float = 0.0`               # 0.0 disables SDPO
- `full_logit_distillation: bool = True`
- `distillation_topk: int | None = 100`
- `distillation_add_tail: bool = True`
- `alpha: float = 0.5`                    # JSD by default
- `is_clip: float | None = 2.0`
- `teacher_regularization: Literal["ema", "trust-region"] = "ema"`
- `teacher_update_rate: float = 0.05`     # EMA rate

### 7.3 EMA teacher

Mirror SDPO. On the policy worker, hold a `teacher_module` that is a
parameter-EMA of the actor. After every optimizer step, call:

```python
with torch.no_grad():
    for p_t, p_s in zip(teacher_module.parameters(), actor_module.parameters()):
        p_t.data.mul_(1 - rate).add_(p_s.data, alpha=rate)
```

Initialize `teacher_module` as a deep copy of `actor_module` at trainer start.
On checkpoint resume, restore both. The teacher must have the same FSDP
sharding as the actor — copy ECHO's ray-remote worker pattern.

## 8 · Slime upstream patch

Your patch shape mirrors `echo-rl/patches/skyrl_minimal_hooks.patch`. It is
~80–120 lines of additive changes. It must do the following:

1. **`Experience.extras: dict[str, Any] | None`** — carry per-sample auxiliary
   tensors (`env_loss_mask`, `sd_target_mask`, `teacher_input_ids`,
   `teacher_attention_mask`, `student_target_offset`, `teacher_target_offset`,
   `target_length`, `full_observation_count`, `world_warning_mask` if you
   add one later). Update `to_device` and `pin_memory` to recurse into it.

2. **`zero_pad_keys` metadata on the training batch** — when Slime pads
   shorter sequences in a batch up to the longest, padding rows of
   `env_loss_mask` and `sd_target_mask` must be zeros, not loss-mask defaults.
   Same one-line trick as ECHO's patch (loss_mask special-case extended to
   any key in `zero_pad_keys`).

3. **`get_aux_policy_loss_context(data) -> dict`** hook on the policy worker.
   Default returns `{}`. Used to compute denominators (for token
   normalization across micro-batches) once per outer batch.

4. **`compute_aux_policy_loss(*, action_log_probs, policy_loss, experience,
   loss_config, microbatch_weight, grad_sum_correction_factor,
   aux_policy_loss_context) -> tuple[Optional[Tensor], dict]`** hook on the
   policy worker. Default returns `(None, {})`. Called inside the
   forward-backward micro-batch path; if the returned tensor is non-None it
   gets added to `loss` before backward.

5. **Pass-through of unknown batch fields into `Experience.extras`** —
   Slime's `BatchIterator.batch_to_experience` should preserve any tensor
   field not in its known list (mirror ECHO patch's `worker_utils.py` change).

If any of these mechanisms already exist in Slime, use them. Re-implement
what's missing.

### Subclassing the Slime worker

Create a thin subclass of Slime's policy worker (call it
`PolarPolicyWorker`). Override:

- `get_aux_policy_loss_context`: compute env-CE token denominator (if
  `loss_reduction == "token_mean"`, accumulate `env_loss_mask.sum()` across
  the whole minibatch; this is the SkyRL pattern from
  `EchoFSDPPolicyWorkerBase`).

- `compute_aux_policy_loss`: see §9.

## 9 · Worker forward (the only really subtle piece)

Inside the policy worker's micro-batch forward, you currently have the
student forward producing `action_log_probs` with shape `[B, T_student]`.

### 9.1 ECHO branch

```python
env_mask = experience.extras.get("env_loss_mask")          # [B, T_student], 0/1
warning_mask = experience.extras.get("world_warning_mask") # optional, can be None
env_only_mask = experience.extras.get("world_env_mask")    # optional, can be None
full_obs_count = experience.extras.get("full_observation_count")   # [B]

echo_loss, echo_metrics = compute_world_model_loss(
    action_log_probs=action_log_probs,
    world_loss_mask=env_mask,
    config=loss_config,                                # carry world_model_coeff etc.
    warning_mask=warning_mask,
    env_mask=env_only_mask,
    full_observation_count=full_obs_count,
    token_normalization_denominator=context.get("world_token_normalization_denominator"),
)
```

If `env_mask is None` or `world_model_coeff <= 0`, skip.

### 9.2 SDPO branch

This is the tricky alignment.

```python
teacher_input_ids = experience.extras.get("teacher_input_ids")
if teacher_input_ids is None or sdpo_coeff <= 0:
    sdpo_loss = None
else:
    teacher_inputs = {
        "input_ids": teacher_input_ids,
        "attention_mask": experience.extras["teacher_attention_mask"],
        "position_ids": compute_position_id_with_mask(experience.extras["teacher_attention_mask"]),
    }
    teacher_module = self.teacher_module          # the EMA copy
    with torch.no_grad():
        teacher_outputs = self._forward_micro_batch(
            teacher_inputs,
            temperature=temperature,
            calculate_entropy=False,
            return_all_logps=full_logit_distillation and not distillation_topk,
            distill_topk=distillation_topk,
            module=teacher_module,
        )
    teacher_log_probs = teacher_outputs["log_probs"]    # [B, T_teacher]

    # --- ALIGNMENT ---
    # Slice both forwards down to the turn-k positions only.
    student_off = experience.extras["student_target_offset"]      # [B]
    teacher_off = experience.extras["teacher_target_offset"]      # [B]
    target_len  = experience.extras["target_length"]              # [B]

    # Per-sample slicing (different lengths per sample → use a packed mask).
    # Easiest: build a [B, max_target_len] gather of student_log_probs and
    # teacher_log_probs, plus a presence mask of shape [B, max_target_len].
    # Then call compute_self_distillation_loss with response_mask=presence
    # and self_distillation_mask=ones_like(presence). The function aggregates
    # per-token, so as long as both tensors are aligned at gathered positions
    # the loss is correct.

    sdpo_loss, sdpo_metrics = compute_self_distillation_loss(
        student_log_probs=student_gathered,
        teacher_log_probs=teacher_gathered,
        response_mask=presence_mask,
        self_distillation_config=loss_config,
        old_log_probs=old_log_prob_gathered,
        student_all_log_probs=student_all_gathered,
        teacher_all_log_probs=teacher_all_gathered,
        student_topk_log_probs=student_topk_gathered,
        teacher_topk_log_probs=teacher_topk_gathered,
        self_distillation_mask=torch.ones_like(presence_mask),
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=None,
    )
```

**Crucial detail.** SDPO's loss function returns the loss already weighted by
`response_mask * self_distillation_mask`. We use `response_mask` as the
"valid sample × valid position" mask (1 where a token actually exists for
that sample's turn-k window) and `self_distillation_mask` as all-ones. This
keeps SDPO's existing reduction code unchanged.

### 9.3 Total loss

```python
aux = 0.0
if echo_loss is not None:
    aux = aux + echo_coeff * echo_loss
if sdpo_loss is not None:
    aux = aux + sdpo_coeff * sdpo_loss

return (aux if isinstance(aux, torch.Tensor) else None), {**echo_metrics, **sdpo_metrics}
```

The Slime hook then adds this to the main loss and backprops once.

### 9.4 EMA update

After the optimizer step, call:

```python
self._ema_update_teacher(rate=teacher_update_rate)
```

(Implementation in §7.3.)

## 10 · Configuration surface

One YAML block, default-off:

```yaml
trainer:
  algorithm:
    # ECHO
    world_model_coeff: 0.0
    world_loss_normalization: full_observation_tokens
    world_model_coeff_schedule: constant
    loss_reduction: token_mean

    # SDPO targeted
    sdpo_coeff: 0.0
    full_logit_distillation: true
    distillation_topk: 100
    distillation_add_tail: true
    alpha: 0.5
    is_clip: 2.0
    teacher_regularization: ema
    teacher_update_rate: 0.05

slime_bridge:
  sdpo:
    enabled: false
    success_threshold: 1.0
    hint_source: deterministic
    max_hint_chars: 2048
    max_teacher_seq_len: 16384

  echo:
    enabled: false      # toggles whether bridge/builder produce env_loss_mask
```

Two toggles to verify ablations:

- `world_model_coeff > 0 + sdpo_coeff = 0` → ECHO only.
- `world_model_coeff = 0 + sdpo_coeff > 0` → SDPO only.
- Both > 0 → joint. Sanity-check loss magnitudes early; SDPO can dominate
  if coeffs aren't normalized to comparable scales. Log
  `world_loss_scaled / policy_loss` and `sdpo_loss / policy_loss` ratios
  every step (mirror ECHO's `world_policy_loss_ratio` metric).

## 11 · Validation plan

**Unit tests** (Polar):
- `tests/trajectory/test_prefix_merging_with_env.py` — see §5.
- `tests/trajectory/test_trace_schema.py` — round-trip `Trace` with
  `env_loss_mask` and `turn_offsets`; assert validation rejects mismatched
  lengths and non-binary mask values.

**Unit tests** (bridge):
- `tests/slime_bridge/test_sdpo_splice.py` — fixture: a two-turn trace,
  hint at turn 1. Assert:
  - `teacher_input_ids[-target_length:] == student_response_ids[s:e]`.
  - `sd_target_mask` sum == `target_length` and all ones are at the tail.
  - `student_target_offset` and `teacher_target_offset` are correct.

**Smoke test** (end-to-end, no GPU):
- A pytest that mocks the trainer side: build a `Sample` with both extras,
  run a fake forward that returns deterministic logits, confirm both losses
  produce the expected scalar (compute by hand on a 4-token window).

**Numerical sanity** (one GPU, calculator example):
- Run vanilla GRPO baseline → confirm reward curve matches existing runs.
- Set `world_model_coeff = 0.5`, leave SDPO off → confirm ECHO loss drops
  monotonically over the first ~50 steps and policy reward isn't worse.
- Enable SDPO with a deterministic hint extractor → confirm `sdpo_loss`
  metric is non-zero on failed sessions, zero on successful ones, and
  `sd_empty_target_batch` ratio is reasonable (< 0.5 typically).

## 12 · Order of operations

Strictly in this order. Each step independently testable.

1. **Schema**: add `env_loss_mask`, `turn_offsets`, `full_observation_count`
   to `Trace` + validators + tests. Existing builders still work because
   these default to empty.
2. **Builder**: `prefix_merging_with_env`. Tests in §5. Wire into registry.
   Existing examples can opt in via `builder.strategy: prefix_merging_with_env`.
3. **Bridge — ECHO half**: forward `env_loss_mask` and
   `full_observation_count` onto Slime samples. No SDPO yet. Sample fixture
   test.
4. **Slime upstream patch**: `extras`, `zero_pad_keys`, `get_aux_policy_loss_context`,
   `compute_aux_policy_loss`. Land this *before* writing aux losses so the
   hooks exist.
5. **ECHO loss**: lift `compute_world_model_loss` and the schedule helper.
   Subclass the Slime worker. Confirm the calculator example trains with
   `world_model_coeff = 0.0` identical to vanilla; then bump it up.
6. **Bridge — SDPO half**: deterministic hint extractor + teacher splicer +
   sample fields. Tests.
7. **SDPO loss**: lift `compute_self_distillation_loss`. Add EMA teacher
   module to the worker. Wire alignment gather. Run the calculator with
   `sdpo_coeff = 0.0` first → must equal step 5 baseline.
8. **Turn it on**: `sdpo_coeff = 0.5` at small scale. Watch metrics.
9. **Optional LLM critic** for hints: only after deterministic-hint version
   trains stably.

## 13 · Hard rules

- **Never decode → re-encode sampled tokens.** This breaks BPE canonicality
  and silently corrupts logprobs. Always tokenize message-level deltas via
  `apply_chat_template` and slice from the delta.
- **Both auxiliary masks live at length `len(response_ids)`.** Same length
  contract as `loss_mask`. Validators must reject otherwise. Padding to
  `T_max` happens in the trainer; pad with zeros for both auxiliary masks.
- **Loss masks are mutually exclusive between assistant and env tokens.**
  At every position, `loss_mask + env_loss_mask <= 1`. SDPO's `sd_target_mask`
  is a *subset* of `loss_mask` (you only KL on student-generated tokens).
  Add an assertion in tests.
- **Teacher forward must be `torch.no_grad()`.** EMA teacher should not
  receive gradient. The student forward provides all the gradient that
  flows into the actor.
- **SDPO records may be missing for a sample.** Loss must gracefully handle
  zero-target batches (return scalar 0 with attached metric, do not crash).
  Mirror SDPO's `empty_target_batch` metric.
- **Default everything off.** A run with `world_model_coeff = 0` and
  `sdpo_coeff = 0` and bridge `echo.enabled = false`, `sdpo.enabled = false`
  must produce numerically identical gradients to the pre-change baseline.

## 14 · Out of scope

- Step-wise per-turn rewards. We are still GRPO over outcome reward at the
  trajectory level. ECHO and SDPO add per-token gradient density without
  changing the outcome-reward signal.
- Re-rolling the teacher trajectory. The teacher is a single likelihood pass
  over the student's tokens with hint context. No second runtime, no second
  agent run.
- Multi-hint per trace. v1 supports exactly one `(turn_index, hint_text)`
  per trace. If a hint is needed at multiple turns, run two SDPO records on
  separate samples or extend later.
- Custom hint distillation across teachers. EMA-of-self only. No external
  teacher model.

## 15 · Files you will create or modify

Polar:
- `src/polar/trajectory/models.py`                              (modify)
- `src/polar/trajectory/builder/prefix_merging_with_env.py`     (create)
- `src/polar/trajectory/registry.py`                            (modify)
- `tests/trajectory/test_prefix_merging_with_env.py`            (create)
- `tests/trajectory/test_trace_schema.py`                       (modify)

Bridge:
- `src/slime_bridge/adapter.py`                                 (modify)
- `src/slime_bridge/sdpo.py`                                    (create)
- `src/slime_bridge/rollout.py`                                 (modify)
- `src/slime_bridge/config.py`                                  (modify)
- `tests/slime_bridge/test_sdpo_splice.py`                      (create)

Trainer (Slime + extension package):
- A patch file analogous to `echo-rl/patches/skyrl_minimal_hooks.patch`     (create)
- `<your_extension>/aux_losses.py` — both losses and EMA helper             (create)
- `<your_extension>/policy_worker.py` — Slime worker subclass               (create)
- Trainer config schema additions                                            (modify)

You should not need to touch `polar.gateway`, `polar.runtime`,
`polar.platform`, or `polar.rollout` core. The proxy already captures the
token ids and logprobs both losses depend on.

## 16 · Acceptance criteria

A run satisfying all of:

- Calculator example passes with `world_model_coeff = 0, sdpo_coeff = 0` and
  reward curve matches the pre-change baseline within numerical noise.
- Calculator example passes with `world_model_coeff = 0.5, sdpo_coeff = 0`
  and `world_loss_unscaled` decreases monotonically over the first 50 steps.
- Calculator example with deterministic hints and `sdpo_coeff = 0.3` shows
  non-zero `sdpo_loss` on at least 25% of steps and final reward at or above
  the GRPO baseline.
- All new pytest tests pass.
- No regressions in existing pytest suite.
