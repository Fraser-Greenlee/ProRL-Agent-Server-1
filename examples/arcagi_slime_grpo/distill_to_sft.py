#!/usr/bin/env python3
"""Convert harvested distillation trajectories into Qwen3.6-tokenized SFT records.

Input: the harvester's jsonl (distill_harvest.py output) — one line per kept
trajectory with OpenAI-style `messages` (assistant turns carry `reasoning_content`
+ `tool_calls`; tool turns carry `content`).

For each trajectory we:
  1. Render the full conversation with Qwen3.6's own chat template via
     `apply_chat_template(..., preserve_thinking=True)` — so SFT is byte-identical
     to what SGLang serves, INCLUDING chain-of-thought on every assistant turn
     (preserve_thinking is belt-and-suspenders; verified to keep all turns here).
  2. Tokenize, and build a loss MASK that supervises ONLY assistant spans
     (system/user/tool tokens get label -100). We locate assistant spans by
     re-rendering the conversation prefix up to and including each assistant turn
     and diffing token boundaries — robust to the template's exact byte layout.
  3. Emit {input_ids, labels, n_tokens, task_id, mode, reward} and report a length
     histogram so the SFT context / context-parallel config can be chosen.

Usage (needs the Qwen3.6-27B tokenizer locally or on the cluster):
  uv run --with transformers --with tokenizers --with jinja2 \
    python distill_to_sft.py --in runs/distill_full/sft_all_correct.jsonl \
      --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/<snap> \
      --out runs/distill_full/sft_qwen.jsonl
Add --profile-only to just print the length histogram without writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_IGNORE = -100


def _load(jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in open(jsonl) if l.strip()]


def _assistant_indices(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def _normalize_messages(messages: list) -> list:
    """Reshape captured OpenHands messages into what the Qwen template expects.

    Captured tool_calls are FLAT {id, name, arguments:<json-string>}; the template
    wants tool_call.function.{name, arguments:<dict>} and calls `arguments|items`
    (so arguments MUST be a mapping, not a string). We parse the JSON string and
    wrap under `function`. Also drop OpenHands-internal keys (responses_item_id,
    origin) that aren't part of the chat protocol.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            norm_tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or tc
                name = fn.get("name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {"_raw": args}  # keep payload rather than crash
                if not isinstance(args, dict):
                    args = {"_value": args}
                norm_tcs.append({
                    "type": "function",
                    "id": tc.get("id"),
                    "function": {"name": name, "arguments": args},
                })
            m["tool_calls"] = norm_tcs
        out.append(m)
    return out


def _encode_with_mask(tok, messages: list) -> tuple[list[int], list[int]]:
    """Tokenize the full conversation once and supervise only assistant spans.

    The Qwen template emits each assistant turn as
        <|im_start|>assistant\\n ... <|im_end|>\\n
    (tool calls render INSIDE that span, before <|im_end|>). We tokenize the full
    rendered text, then walk the token stream: the tokens from each
    `<|im_start|>assistant` header up to and including the next `<|im_end|>` are
    the assistant turn — we supervise those, mask everything else (-100).

    Single full render (no O(N^2) prefix re-rendering, no prefix-render edge cases
    like a bare [system] prefix raising "No user query found").
    """
    full = tok.apply_chat_template(messages, tokenize=False, preserve_thinking=True)
    input_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [LABEL_IGNORE] * len(input_ids)

    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # "assistant" header token(s) right after <|im_start|>. Encode without specials.
    asst_hdr = tok("assistant", add_special_tokens=False)["input_ids"]

    n = len(input_ids)
    i = 0
    while i < n:
        if input_ids[i] == im_start and input_ids[i + 1 : i + 1 + len(asst_hdr)] == asst_hdr:
            # Assistant turn: supervise from just after the header through <|im_end|>.
            j = i + 1 + len(asst_hdr)
            while j < n and input_ids[j] != im_end:
                labels[j] = input_ids[j]
                j += 1
            if j < n:  # include the closing <|im_end|> so the model learns to stop
                labels[j] = input_ids[j]
            i = j + 1
        else:
            i += 1
    return input_ids, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", type=Path, required=True,
                    help="harvested SFT jsonl (distill_harvest.py output)")
    ap.add_argument("--tokenizer", required=True, help="path/name of Qwen3.6-27B tokenizer")
    ap.add_argument("--out", type=Path, help="output tokenized SFT jsonl")
    ap.add_argument("--profile-only", action="store_true",
                    help="just print the length histogram; don't tokenize-and-write")
    ap.add_argument("--max-len", type=int, default=None,
                    help="drop (and report) trajectories longer than this many tokens")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if not tok.chat_template:
        raise SystemExit("tokenizer has no chat_template")

    rows = _load(args.in_jsonl)
    print(f"loaded {len(rows)} trajectories from {args.in_jsonl}")

    out_f = None
    if args.out and not args.profile_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out, "w")

    lengths = []
    n_written = 0
    n_dropped_len = 0
    sup_fracs = []
    for r in rows:
        msgs = r.get("messages") or []
        if not msgs:
            continue
        msgs = _normalize_messages(msgs)
        try:
            input_ids, labels = _encode_with_mask(tok, msgs)
        except Exception as e:
            print(f"  WARN {r.get('task_id')}/{r.get('mode')}: encode failed: {e}")
            continue
        n = len(input_ids)
        lengths.append(n)
        sup = sum(1 for x in labels if x != LABEL_IGNORE)
        sup_fracs.append(sup / n if n else 0.0)

        if args.max_len is not None and n > args.max_len:
            n_dropped_len += 1
            continue
        if out_f is not None:
            out_f.write(json.dumps({
                "task_id": r.get("task_id"), "mode": r.get("mode"),
                "reward": r.get("reward"), "compression_pct": r.get("compression_pct"),
                "n_tokens": n, "n_supervised": sup,
                "input_ids": input_ids, "labels": labels,
            }) + "\n")
            n_written += 1
    if out_f:
        out_f.close()

    # ── Length histogram → informs SFT context / context-parallel sizing ──
    if lengths:
        lengths.sort()
        import statistics
        def pct(p): return lengths[min(len(lengths) - 1, int(len(lengths) * p))]
        print("\n=== trajectory token-length distribution ===")
        print(f"  n={len(lengths)} min={lengths[0]} mean={statistics.mean(lengths):.0f} "
              f"median={pct(0.5)} p90={pct(0.9)} p95={pct(0.95)} p99={pct(0.99)} max={lengths[-1]}")
        for thr in (16384, 32768, 49152, 65536, 98304, 131072):
            over = sum(1 for x in lengths if x > thr)
            print(f"  > {thr:>6} tok: {over:4d} ({100*over/len(lengths):.1f}%)")
        print(f"  mean supervised (assistant) fraction: {statistics.mean(sup_fracs):.2f}")
    if out_f is not None or args.out:
        print(f"\nwrote {n_written} SFT records"
              + (f" ({n_dropped_len} dropped > --max-len {args.max_len})" if args.max_len else "")
              + (f" -> {args.out}" if args.out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
