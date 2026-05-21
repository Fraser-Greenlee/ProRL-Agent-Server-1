# Polar Platform — Implementation Plan

This document specifies the Polar platform frontend (observability + task
submission). It is the source
of truth for the implementation agent. Every design decision below has been
reviewed; treat them as fixed unless the document says otherwise.

---

## 1. Context

Polar is a distributed rollout orchestration framework for agent training and
evaluation. The codebase lives at `/raid/binfeng/workspace/ProRL-Agent-Server`.

### 1.1 Components in the current system

| Component | Path / Port | Role |
|---|---|---|
| Rollout server | [src/polar/rollout/server.py](../src/polar/rollout/server.py), port `8080` | Central orchestrator. Accepts task submissions, dispatches sessions to gateway nodes, collects results. FastAPI + uvicorn. |
| Gateway nodes | [src/polar/gateway/server.py](../src/polar/gateway/server.py), ports `8100`, `8101` | Per-node executor. Runs three worker pools (init / run / postrun), proxies LLM API calls (OpenAI / Anthropic / Google) to SGLang, builds + evaluates trajectories. |
| SGLang backends | external, ports `8000`, `8001` | LLM inference (OpenAI-compatible). One per GPU group. |
| Runtimes | Docker / Apptainer containers | Sandbox where harnesses run. Defined by [RuntimeSpec](../src/polar/runtime/models.py). |
| Harnesses | `examples/<task>/<harness_name>/` | Agent CLIs (e.g. `claude_code`, `codex`, `gemini_cli`, `opencode`, `pi`, `qwen_code`, `swe_agent`). |
| CLI | [src/polar/cli.py](../src/polar/cli.py) | `polar serve_rollout`, `polar serve_gateway`, `polar submit`, `polar status`. |

### 1.2 Existing HTTP routes

**Rollout server** (`src/polar/rollout/server.py`):

- `POST /rollout/task/submit` — submit `TaskRequest`, returns `{task_id, status: "running"}`.
- `GET /rollout/task/{task_id}` — `TaskStatus`: task progress + per-session results + result file paths.
- `GET /rollout/status` — pipeline + node summary.
- `GET /nodes`, `GET /nodes/{node_id}`, `DELETE /nodes/{node_id}` — node registry.
- `POST /nodes/register`, `POST /nodes/{node_id}/heartbeat` — used by gateways.
- `POST /callbacks/session_result` — gateways push terminal results here.
- `GET /health`.

**Gateway server** (`src/polar/gateway/server.py`):

- `POST /sessions` — create / dispatch session.
- `GET /sessions/{session_id}` — `SessionStatusResponse`.
- `DELETE /sessions/{session_id}` — cancel.
- `POST /{path:path}` — LLM proxy (auto-detects OpenAI / Anthropic / Google; transforms; streams).
- `GET /v1/models` — proxies SGLang model list.
- `GET /health` — node metrics: `active_status_counts`, `available_init_workers`, `available_run_workers`, `available_postrun_workers`, `active_sessions`.
- `GET /admin/sglang/status`, `POST /admin/sglang/pause`, `POST /admin/sglang/resume`.

### 1.3 Data schemas (already defined — reuse, do not redefine)

- `TaskRequest` — [src/polar/rollout/models.py](../src/polar/rollout/models.py) lines 59–71. Fields: `task_id`, `instruction`, `num_samples`, `timeout_seconds`, `runtime` (`RuntimeSpec`), `agent` (`AgentSpec`), `builder` (`StrategySpec`), `evaluator` (`EvaluatorSpec`), `callback_url`, `metadata`.
- `SessionStatus` enum — same file. States: `REGISTERED` → `INITIALIZING` → `READY` → `RUNNING` → `POST_RUN` → `BUILDING` → `EVALUATING` → `COMPLETED` | `ERROR` | `TIMEOUT`.
- `SessionResult` — same file, lines 120–130. Fields: `session_id`, `task_id`, `status`, `trajectory`, `timing` (`SessionTiming`), `node_id`, `error`, `metadata`.
- `SessionTiming` — keys actually seen on disk: `register_to_init_queue_ms`, `init_ms`, `run_ms`, `postrun_ms`. (`eval_ms`, `build_ms` may appear in `metadata` instead — check.)
- `Trajectory`, `Trace`, `CompletionSession`, `CompletionRecord` — [src/polar/trajectory/models.py](../src/polar/trajectory/models.py). `Trace` has `prompt_ids`, `response_ids`, `prompt_messages`, `response_messages`, `reward`, `finish_reason`, `response_logprobs`, `loss_mask`, `metadata`.
- `RuntimeSpec` — [src/polar/runtime/models.py](../src/polar/runtime/models.py) lines 56–90.
- `AgentSpec` — [src/polar/agent/models.py](../src/polar/agent/models.py) lines 32–66.
- Topology config — [src/polar/config/topology.py](../src/polar/config/topology.py); example [examples/calculator/topology.yaml](../examples/calculator/topology.yaml).

### 1.4 What is persisted today

- `<save_dir>/<task_id>/ses_<session_id>.json` — terminal `SessionResult` per session, written by `Pipeline` when a session ends. The default save_dir is `./rollout_results` (set per-topology). Real examples currently in `rollout_results/task_calculator-<harness>-<ts>/ses_sk-polar-<uuid>.json`.
- `examples/<task>/batches/<ts>/<harness>/{request,response,summary}.json` — written by submission scripts, not Polar.
- **Nothing else.** No database. All in-memory state (active sessions, gateway `SessionStore`, `RolloutManager` task table) is lost on restart. `polar status` only reads live HTTP.

### 1.5 Gaps this project fixes

1. No web UI. `polar status` is CLI text and may not even reach gateways in some configs.
2. Submissions require hand-edited JSON files run through Python scripts. No prebuilt picker.
3. No endpoint lists tasks or active sessions (`/rollout/task/{id}` only works if you already have the id).
4. Raw client request and transformed SGLang request live only in gateway memory (`SessionStore`, [src/polar/gateway/storage.py](../src/polar/gateway/storage.py)) during the session and are cleared on completion. Built trajectory traces survive but the original API payloads do not.
5. No push channel — everything is polling.
6. No topology visualization. Operators have no map of which gateway talks to which SGLang.

---

## 2. Goals and confirmed design decisions

### 2.1 Functional goals

1. Submit a rollout task from a browser using a template-driven form (`calculator`, `swebench`, future templates) with dropdowns / fields for harness, model, runtime, builder, evaluator, sample count, timeout.
2. Visualize the live topology (rollout, gateways, SGLang backends, active runtimes) with worker-pool occupancy.
3. Browse tasks (live and historical) and their per-session results.
4. Drill into a session and see, across stages: raw client request, transformed SGLang request, response, reconstructed trajectory traces, evaluator output.
5. Cancel an in-flight session.
6. Smooth UX: live updates without manual refresh, fast page loads, sensible defaults, clear error states.

### 2.2 Decisions (locked)

| Decision | Choice |
|---|---|
| Service shape | **Separate process** `polar serve_platform` (new CLI subcommand). Does not share fate with rollout server. |
| Polar-side changes | **Add small read-only endpoints** to rollout and gateway. Also add SSE push (`/events`) on both. |
| Templates | **Curated JSON templates** checked into the repo. |
| History | **Scan `rollout_results/`**; no SQLite or Postgres. In-memory index in platform service. |
| Frontend stack | **React + Vite + Tailwind + shadcn/ui**, with **React Flow** for topology. TanStack Query for data, TanStack Router for routing. |
| Topology fidelity | **Full graph**: rollout → gateways → SGLang → active runtimes. Worker-pool bars per node. |
| Write surface | **Submit tasks + cancel sessions.** No admin pause/resume / drain in v1. |
| Access | **Local-only, no auth.** Bind to `127.0.0.1`. |
| Completion records | **Persist to disk during the run** (gateway streams each `CompletionRecord` to a file) so they remain viewable after completion. |
| v1 scope | **Full vision**: includes SSE push, charts, filters, batch comparison. Delivered in milestones M1–M6. |

### 2.3 Non-goals (explicit)

- Multi-user auth, sharing, or RBAC.
- Storing payloads in a relational database.
- Replacing the submission Python scripts (they continue to work).
- Modifying training-time behavior. The platform service must not slow down rollouts measurably.
- Mobile-friendly layout.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Browser  (React SPA, served by platform service)          │
│   Dashboard · Submit · Tasks · Task detail · Session detail    │
└──────────────────────────┬─────────────────────────────────────┘
                           │ HTTP + SSE  (default port 8090)
┌──────────────────────────▼─────────────────────────────────────┐
│ polar serve_platform  (new FastAPI service)               │
│  /api/topology · /api/tasks · /api/sessions · /api/events      │
│  /api/templates · POST /api/submit · GET /api/models           │
│  Static mount of web/dist/ at /                                │
└──┬─────────────────┬───────────────────────────┬───────────────┘
   │ HTTP polling    │ HTTP polling              │ FS scan + watchdog
┌──▼──────────┐ ┌────▼────────────┐ ┌────────────▼───────────────┐
│ Rollout     │ │ Gateway nodes   │ │ <save_dir>/                │
│ :8080       │ │ :8100, :8101    │ │   task_<id>/               │
│ +new routes │ │ +new routes     │ │     ses_<sid>.json         │
│ +SSE        │ │ +SSE            │ │     sessions/<sid>/        │
└─────────────┘ └─────────────────┘ │       completions/NNN.json │
                                    └────────────────────────────┘
```

**Boundaries:**

- The platform service is read-mostly. The only writes it performs against Polar are `POST /rollout/task/submit` (proxied from `/api/submit`) and `DELETE /sessions/{id}` (proxied from `/api/sessions/{id}` DELETE).
- The platform service never connects to SGLang directly. SGLang health surfaces through gateway `/health`.
- Frontend never calls rollout or gateway directly. Single origin = platform service.

---

## 4. Polar-side changes

Keep changes minimal and read-only with one exception: completion-record disk writes on the gateway.

### 4.1 Rollout server changes

File: [src/polar/rollout/server.py](../src/polar/rollout/server.py).

Add routes:

1. `GET /tasks`
   - Query params: `status` (`running` | `completed` | `failed`), `harness`, `since` (ISO timestamp), `limit` (default 200).
   - Returns: `{tasks: [TaskSummary]}` where `TaskSummary` is `{task_id, status, harness, model, num_samples, completed_sessions, created_at, updated_at, save_dir_path}`.
   - Implementation: merge `RolloutManager._tasks` in-memory dict with a scan of `<save_dir>/task_*/` directories. Harness and model are read from each task's first session result (or from a small new index file — see 4.4).

2. `GET /tasks/{task_id}/sessions`
   - Returns: `{sessions: [SessionSummary]}` where `SessionSummary` is `{session_id, status, node_id, reward, timing, error}`.
   - Implementation: combine in-memory session state with on-disk `ses_*.json` files.

3. `GET /events` (SSE, `text/event-stream`)
   - Event types: `task.created`, `task.updated`, `task.completed`, `session.state_changed`.
   - Payloads: `{task_id, session_id?, status, timestamp}`.
   - Implementation: a small in-process pub/sub (asyncio Queue per subscriber) hooked into `RolloutManager` and `Pipeline` callback paths. Drop oldest if a subscriber falls behind.

### 4.2 Gateway server changes

File: [src/polar/gateway/server.py](../src/polar/gateway/server.py).

Add routes:

1. `GET /sessions`
   - Query params: `status` filter, `task_id` filter, `limit`.
   - Returns: `{sessions: [{session_id, task_id, status, created_at, completion_count, model_used}]}`.
   - Implementation: enumerate `SessionRegistry` (and `SessionStore` for completion_count).

2. `GET /sessions/{session_id}/completions`
   - Returns: `{session_id, completions: [CompletionRecordView]}` where `CompletionRecordView` is `{completion_id, timestamp, api_type, model_requested, model_used, original_request, transformed_request, response, transformed_response, error?}`.
   - Implementation: read from in-memory `SessionStore` if active; fall back to on-disk `completions/*.json` (see 4.3) for finished sessions.

3. `GET /events` (SSE) — emits `session.state_changed`, `session.completion_added`. Same pub/sub pattern.

### 4.3 Persist completion records during the run

File: [src/polar/gateway/storage.py](../src/polar/gateway/storage.py).

Change `SessionStore.save_message()`:

- Continue appending to in-memory list as today (no behavior change for callers).
- Enqueue the record to a `CompletionWriter` (one per gateway process, lazy-initialized).
- `CompletionWriter` runs a background `asyncio.Task` that drains a bounded queue (default size 1024) and writes one file per record to:
  `<save_dir>/<task_id>/sessions/<session_id>/completions/<NNNN>-<completion_id>.json`
  where `NNNN` is zero-padded sequence per session.
- The writer must be off the hot path: `save_message()` returns immediately after enqueue. If the queue is full, log a warning and drop the persistence (in-memory copy still exists). Never raise back into the proxy.
- File schema mirrors `CompletionRecord` but adds `original_request` (already on the record) and `transformed_request` if available. Truncate any single field over `max_field_bytes` (default 1 MiB) with a `__truncated: true` marker.
- New topology config knob `gateway.completion_persistence: {enabled: bool, max_field_bytes: int, queue_size: int}`; defaults to enabled.

### 4.4 Optional small index file (nice-to-have, not required for M1)

When a task completes (or on every session result), `Pipeline` writes `<save_dir>/<task_id>/_index.json` with `{task_id, status, harness, model, num_samples, completed_sessions, created_at, updated_at, runtime_image}`. This lets `GET /tasks` return rich summaries without opening every session file. If skipped, platform service does the same indexing in memory.

### 4.5 What must not change

- Existing request/response shapes for `POST /rollout/task/submit`, `GET /rollout/task/{id}`, `POST /sessions`, `DELETE /sessions/{id}`, `POST /{path:path}` proxy. Submission scripts (`examples/*/submit_*.py`) must keep working unmodified.
- Trajectory / evaluator builder registry behavior.
- Any timing or correctness of the LLM proxy. The completion writer must be measured not to add latency on a representative run.

---

## 5. New service: `polar serve_platform`

### 5.1 Repository layout

```
src/polar/platform/
    __init__.py
    cli.py                  # argparse subcommand wiring
    server.py               # FastAPI app, static mount, lifespan
    config.py               # PlatformConfig (port, topology path, save_dir)
    upstream.py             # HTTP clients for rollout + gateways, with retry
    fs_index.py             # rollout_results/ index + watchdog
    sse_fanout.py           # subscribes to upstream SSE streams, fans out to clients
    api/
        __init__.py
        topology.py
        tasks.py
        sessions.py
        submit.py
        templates.py
        events.py
        models.py           # API response Pydantic models (distinct from polar.* internals)
    templates/
        calculator.json
        swebench.json
web/                        # bundled frontend, see section 6
```

### 5.2 CLI subcommand

Register in [src/polar/cli.py](../src/polar/cli.py):

```
polar serve_platform \
    -c examples/calculator/topology.yaml \
    [--host 127.0.0.1] [--port 8090] \
    [--rollout-url http://127.0.0.1:8080] \
    [--save-dir ./rollout_results]
```

Defaults:

- Host `127.0.0.1`, port `8090`.
- `--rollout-url` derived from topology if not provided.
- `--save-dir` derived from topology if not provided.

### 5.3 HTTP API

All routes return JSON unless marked otherwise. All routes are versionless under `/api/`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/topology` | Static topology (from YAML) merged with live `/health` from rollout and each gateway. |
| `GET` | `/api/tasks` | List tasks with filters (`status`, `harness`, `since`, `limit`). Backed by `fs_index` + rollout `/tasks`. |
| `GET` | `/api/tasks/{task_id}` | Single-task detail incl. session summaries. |
| `GET` | `/api/sessions/{session_id}` | Session detail: stage timings, status, error, node, links. |
| `GET` | `/api/sessions/{session_id}/trajectory` | Built trajectory traces from `SessionResult`. |
| `GET` | `/api/sessions/{session_id}/completions` | Completion records. Tries gateway in-memory first, falls back to on-disk files. |
| `GET` | `/api/sessions/{session_id}/evaluation` | Extracted evaluation result (`outcome_reward`, `trace_rewards`, evaluator metadata). |
| `GET` | `/api/models` | Available models, by querying each gateway's `/v1/models` (deduped). |
| `GET` | `/api/templates` | List curated templates. |
| `GET` | `/api/templates/{name}` | One template's payload skeleton + schema hints. |
| `POST` | `/api/submit` | Validate body against `TaskRequest` shape, then proxy to rollout `POST /rollout/task/submit`. Returns the rollout response. |
| `DELETE` | `/api/sessions/{session_id}` | Proxies to the appropriate gateway. |
| `GET` | `/api/events` | SSE multiplex of upstream events. |
| `GET` | `/api/health` | Service health + upstream reachability. |
| `GET` | `/openapi.json`, `/docs` | FastAPI defaults. |
| `GET` | `/*` | Static frontend (SPA fallback). |

### 5.4 Filesystem index

- On startup, walk `<save_dir>/task_*/` once. Read `_index.json` if present, else inspect `ses_*.json` files. Build an in-memory dict `{task_id: TaskSummary}`.
- Watch `<save_dir>/` with `watchdog` (or polling fallback at 2 s) to pick up new files. Re-index incrementally on file events.
- Index supports filtering by `status`, `harness`, time range.
- Memory budget: store summaries only, not full payloads. Open the session file on demand for detail views.

### 5.5 SSE fan-out

- `sse_fanout` opens long-lived `httpx.AsyncClient` streams to `<rollout>/events` and `<each-gateway>/events` on service startup.
- Reconnects with exponential backoff (1 s → 30 s cap) on disconnect.
- Each connected browser client gets an `asyncio.Queue` (bounded, drop-oldest). Server-side filters by route subscription (`?topics=topology,tasks,session:<id>`).

### 5.6 Templates

Schema for `templates/<name>.json`:

```json
{
  "name": "calculator",
  "title": "Calculator (toy end-to-end)",
  "description": "Tiny example: edit calculator.py to make tests pass.",
  "default_harness": "claude_code",
  "supported_harnesses": ["claude_code", "codex", "gemini_cli", "opencode", "pi", "qwen_code", "swe_agent"],
  "task_request_skeleton": { /* full TaskRequest payload with placeholders */ },
  "ui_hints": {
    "editable_fields": ["agent.harness", "agent.model_name", "num_samples", "timeout_seconds", "runtime.gpus"],
    "field_help": { "num_samples": "Number of independent rollouts to run." }
  }
}
```

Two templates ship with v1:

- `calculator.json` — mirrors the payload built by [examples/calculator/submit_calculator_task.py](../examples/calculator/submit_calculator_task.py).
- `swebench.json` — mirrors [examples/swebench_verified/submit_swebench_tasks.py](../examples/swebench_verified/submit_swebench_tasks.py). Includes an `instance_id` placeholder; UI must accept a single instance id for v1 (the swebench script supports batches, but the UI submits one task at a time).

A CI test must validate every template's `task_request_skeleton` against the live `TaskRequest` Pydantic model so templates can't silently rot.

---

## 6. Frontend

### 6.1 Stack

- React 19, TypeScript, Vite.
- Tailwind CSS + shadcn/ui (Radix-based primitives, copy-paste components).
- TanStack Query for data fetching + cache.
- TanStack Router for routing (file-based).
- React Flow for the topology graph.
- Zod for form validation.
- `@microsoft/fetch-event-source` for SSE.
- Vitest for unit tests, Playwright for one end-to-end happy path.

Build output `web/dist/` is served by the FastAPI service. During development, run `vite dev` on port 5173 with a proxy to the platform service.

### 6.2 Layout

```
web/
    package.json
    vite.config.ts
    tailwind.config.ts
    index.html
    src/
        main.tsx
        router.tsx
        api/
            client.ts          # fetch wrapper
            queries.ts         # TanStack Query hooks
            sse.ts             # SSE subscription helper
            types.ts           # generated from /openapi.json
        components/
            Layout.tsx
            NavBar.tsx
            TopologyGraph.tsx
            WorkerPoolBars.tsx
            StageTimeline.tsx
            JsonView.tsx
            CompletionDiff.tsx
            TraceList.tsx
            RewardSpark.tsx
            TaskTable.tsx
            SessionTable.tsx
            StatusPill.tsx
            EmptyState.tsx
        routes/
            __root.tsx
            index.tsx           # Dashboard
            submit.tsx
            tasks/
                index.tsx       # tasks list
                $taskId.tsx     # task detail
            sessions/
                $sessionId.tsx  # session detail (tabs)
        hooks/
            useSSE.ts
            useTopology.ts
            useTaskList.ts
            useSession.ts
        lib/
            time.ts             # human-readable durations
            colors.ts           # status palette
            rewards.ts          # reward formatting and aggregation
```

### 6.3 Routes and views

#### 6.3.1 Dashboard (`/`)

- Top: topology graph (React Flow). Nodes:
  - 1 `rollout` node (center-top) with health pill and address.
  - N `gateway` nodes (one per topology entry). Each shows worker-pool bars (init / run / postrun) with current vs max, and `active_sessions` count.
  - 1 `sglang` node per gateway under it, labeled with model name and base URL.
  - Floating chips below each gateway for currently active session ids (max 5 visible, "+ N more").
- Right column: "Recent tasks" list (last 10), each clickable.
- Bottom: aggregate counters (running tasks, running sessions, completed today).
- Live: subscribes to `/api/events` on topics `topology,tasks`.

#### 6.3.2 Submit (`/submit`)

- Step 1: pick a template (dropdown). Loads `task_request_skeleton`.
- Step 2: form with sections:
  - **Agent**: harness (dropdown from template's `supported_harnesses`), model (dropdown from `/api/models`).
  - **Runtime**: backend (docker | apptainer), image (text, prefilled), cpus, gpus, memory_mb.
  - **Strategies**: builder (dropdown: `per_request`, `prefix_merging`), evaluator (dropdown: `session_completed`, `test_on_output`, `swebench_harness`). Show evaluator-specific config fields when relevant.
  - **Run**: `num_samples`, `timeout_seconds`.
  - **Metadata**: free-form key/value pairs.
  - "Edit raw JSON" toggle reveals the full `TaskRequest` payload in a Monaco editor; toggle off re-syncs back to fields when possible.
- Validate with Zod against a schema mirrored from `TaskRequest`.
- Submit → `POST /api/submit` → on success, navigate to `/tasks/{task_id}`.
- Surface server validation errors inline.

#### 6.3.3 Tasks list (`/tasks`)

- Filter chips: status, harness, time range, "running only".
- Sortable columns: created_at, harness, num_samples, completed_sessions, mean_reward, duration.
- Row click → `/tasks/{task_id}`.
- Live: row badges update from SSE; new rows append on `task.created`.

#### 6.3.4 Task detail (`/tasks/{task_id}`)

- Header: task_id, harness, model, status pill, created/updated, link to runtime image.
- Reward distribution sparkline (`RewardSpark`).
- Sessions table (one row per session): session_id, status, node, reward, total duration, error preview.
- Row click → `/sessions/{session_id}`.
- "Cancel all running" button (iterates `DELETE` calls). Confirm dialog.

#### 6.3.5 Session detail (`/sessions/{session_id}`)

Header: session_id (copy button), task link, status, node, total duration, error if any. "Cancel" button if running.

Tabs:

1. **Timeline** — `StageTimeline` component renders a horizontal Gantt of stages using `timing.*_ms`. Each stage labeled with absolute ms. Hover for exact values.
2. **Completions** — chronological list of `CompletionRecord`s. Each row expands to a 4-panel `CompletionDiff`:
   - Raw client request (original API format)
   - Transformed SGLang request (normalized OpenAI format)
   - SGLang response
   - Transformed response (back to client API format)
   Each panel: language-aware JSON viewer with search, fold, copy. "Copy as cURL" generates the raw client request.
   Live updates: subscribed to `session.completion_added` on SSE.
3. **Trajectory** — list of `Trace`s. Each card:
   - prompt_messages and response_messages (chat-style render with role/content)
   - reward, finish_reason, response_logprobs sparkline if present
   - collapsible token-id view
4. **Evaluation** — `outcome_reward`, per-trace rewards table, evaluator name + config, any captured stdout from the evaluator container in `metadata.evaluation.*`.
5. **Raw JSON** — pretty-printed `SessionResult` plus a link to the on-disk file path.

Polling cadence:

- Inactive tab: 2 s.
- Active tab (Timeline / Completions) for a running session: SSE driven; fallback poll 500 ms if SSE disconnected.
- Completed session: no polling.

### 6.4 Styling and UX rules

- Color tokens via Tailwind theme. Status palette:
  - `RUNNING` blue, `COMPLETED` green, `ERROR` red, `TIMEOUT` amber, `INITIALIZING/BUILDING/EVALUATING` purple, `REGISTERED/READY` gray.
- Long ids (session_id, task_id) always show short prefix + copy button.
- Empty states: clear message + suggested action ("No tasks yet. Submit one →").
- Loading: skeleton screens, never spinners alone.
- Errors: inline with retry, never silent.
- Never block the UI on completion-record fetches; lazy-load per expand.

---

## 7. Real-time updates

- The platform service maintains one upstream SSE connection per Polar service (rollout + each gateway), all opened at startup and reconnected on failure.
- Browser clients connect to a single `/api/events` channel and subscribe to topics via query string. The service fans out filtered events.
- Event payload shape: `{type: string, ts: ISO8601, data: object}`. Types in use:
  - `task.created`, `task.updated`, `task.completed`
  - `session.state_changed` (carries `session_id`, `status`)
  - `session.completion_added` (carries `session_id`, `completion_id`)
  - `node.health_changed`
- Heartbeat events (`type: "ping"`) every 15 s to keep proxies from closing the stream.

---

## 8. Milestones and acceptance criteria

Each milestone is independently shippable. Do not merge a milestone until its acceptance criteria pass.

### M1 — Backbone (no Polar changes)

Scope:

- `polar serve_platform` boots. Static topology + historical browsing.
- FS index over `rollout_results/` with watchdog.
- Routes: `/api/topology`, `/api/tasks`, `/api/tasks/{id}`, `/api/sessions/{id}`, `/api/sessions/{id}/trajectory`, `/api/sessions/{id}/evaluation`, `/api/templates`, `/api/templates/{name}`, `/api/health`.
- Frontend: Dashboard (topology from YAML only, no live worker bars yet), Tasks list, Task detail, Session detail (Timeline + Trajectory + Evaluation + Raw JSON tabs). Submit page non-functional placeholder.

Acceptance:

- With the existing calculator example having already produced results in `rollout_results/`, every existing session can be navigated to and rendered correctly.
- New session files appearing on disk show up in the UI within 5 s without manual refresh.
- `polar serve_platform` exits cleanly on SIGINT.

### M2 — Live state (small rollout/gateway endpoints)

Scope:

- Add `GET /tasks`, `GET /tasks/{id}/sessions` to rollout server.
- Add `GET /sessions` to gateway server.
- Wire `/api/tasks` and topology to merge live data.
- Dashboard worker-pool bars and active-session chips fed by gateway `/health` + `/sessions`.

Acceptance:

- During a running calculator submission, the dashboard shows worker pools changing and active sessions appearing/disappearing live (polled at 2 s).
- Pre-existing tasks still appear, sourced from FS index.

### M3 — Submit form

Scope:

- `POST /api/submit` and `GET /api/models` on platform service.
- Calculator and swebench templates checked in.
- Submit page fully functional: template picker, dropdowns, raw JSON toggle, validation.
- CI test validates templates against `TaskRequest`.

Acceptance:

- Submitting calculator from the UI completes successfully end-to-end against a local Polar stack; result visible in Tasks list and Session detail.
- Submitting swebench (single instance) completes the same way.
- Invalid payloads (bad harness, missing fields) show inline errors and do not hit the rollout server.

### M4 — Mid-flight completion payloads

Scope:

- Gateway `SessionStore.save_message` queues writes to `completions/NNNN-<id>.json`.
- Gateway `GET /sessions/{id}/completions` route.
- Platform service `/api/sessions/{id}/completions` (memory first, disk fallback).
- Session detail Completions tab with 4-panel `CompletionDiff` viewer.

Acceptance:

- For a running session, completions appear in the tab as they happen, with original and transformed payloads visible.
- For a completed session, completions can still be opened from disk.
- Benchmark: enabling completion persistence does not raise per-completion proxy latency by more than 1 ms p99 on a synthetic loopback test.

### M5 — Push updates

Scope:

- `GET /events` SSE on rollout and gateway.
- Platform service fan-out via `/api/events`.
- Frontend subscribes per route; falls back to polling on disconnect.

Acceptance:

- With 5 sessions running, the dashboard updates without polling activity in DevTools (single SSE connection per client).
- Killing one gateway briefly does not crash the platform service; UI shows the gateway as unreachable and recovers when it returns.

### M6 — Polish

Scope:

- Reward distribution charts on Task detail.
- Tasks list filters: harness, status, time range, "running only".
- Batch comparison view: pick two tasks → side-by-side reward stats, timing stats, error breakdown. New route `/compare?a=...&b=...`.
- Empty states, loading skeletons, keyboard shortcuts (`/` to focus search, `g t` to go to tasks).

Acceptance:

- All routes have non-broken empty states.
- Comparing two completed calculator runs renders without UI errors.

### M7 — Cancel

Scope:

- `DELETE /api/sessions/{id}` proxy.
- Cancel button on Session detail and "Cancel all running" on Task detail with confirm dialog.

Acceptance:

- Cancelling a running session transitions it to `ERROR` (or whatever rollout reports) in < 5 s and is reflected in the UI.

---

## 9. Testing strategy

### 9.1 Backend

- pytest with FastAPI `TestClient` for every new route on rollout, gateway, and platform.
- Unit tests for `fs_index` (creation, update, delete, watchdog event handling).
- Unit test for `CompletionWriter`: ordered writes, queue overflow drops, truncation of oversize fields, no exception bubbling.
- Regression test: a synthetic 1000-completion run with persistence enabled vs disabled; assert p99 latency delta < 1 ms.
- Schema test: every `templates/*.json` must validate against `TaskRequest`. Lives in `tests/platform/test_templates.py`.

### 9.2 Frontend

- Vitest for components: `StageTimeline`, `CompletionDiff`, `TopologyGraph`, `JsonView`, form validation, status palette.
- React Testing Library for route smoke tests with mocked `/api`.
- One Playwright happy-path test: start a local Polar stack via the existing calculator scripts, run `polar serve_platform`, navigate Dashboard → Submit → Tasks → Session detail (Timeline + Completions + Trajectory).

### 9.3 Manual checklist before declaring v1 done

1. Submit a calculator task across all 7 harnesses; UI shows each session detail correctly.
2. Submit one swebench instance; UI handles the larger Docker image and longer runs.
3. Restart the rollout server mid-run; platform service shows the unreachable banner and recovers when the server returns. Historical tasks remain browsable.
4. Kill one gateway; UI marks the node unhealthy. Running sessions on that gateway transition to `ERROR` (per existing Polar behavior) and show up in the UI.
5. Open Session detail for an `ERROR` session; error message is visible and copyable.
6. Compare two completed calculator runs in the compare view.

---

## 10. Operational notes

- Default ports: rollout `8080`, gateways `8100`+, SGLang `8000`+, **platform `8090`**. Keep this free.
- Run order during development:
  1. SGLang servers.
  2. `polar serve_rollout`.
  3. `polar serve_gateway` per node.
  4. `polar serve_platform -c <topology.yaml>`.
- Logs: stdout only, structured via `polar.logging`. No file logging in v1.
- Frontend dev server: `cd web && pnpm dev`. Production: `pnpm build` produces `web/dist/`, copied into the wheel.

---

## 11. Open considerations (track but do not block v1)

1. **Template drift** — if `submit_*.py` payloads diverge from the curated templates, users will be confused. The CI schema test catches structural drift; semantic drift (e.g., a new evaluator config field) requires manual template updates. Document this in `src/polar/platform/templates/README.md`.
2. **Disk pressure** — a high-volume run may generate many completion files. Mitigations already specified: per-field truncation, queue cap, config knob to disable. If this proves a real problem, switch to a single JSONL file per session in a follow-up.
3. **SSE through reverse proxies** — not relevant for local-only v1, but document the `X-Accel-Buffering: no` and chunked-transfer headers we already emit so future remote deploys work.
4. **Authentication** — once anyone wants to expose this beyond `127.0.0.1`, add a static bearer token read from `topology.yaml` and middleware that requires it on `/api/*`. Out of scope for v1.

---

## 12. File-by-file change index (for the implementing agent)

New files:

- `src/polar/platform/__init__.py`
- `src/polar/platform/cli.py`
- `src/polar/platform/server.py`
- `src/polar/platform/config.py`
- `src/polar/platform/upstream.py`
- `src/polar/platform/fs_index.py`
- `src/polar/platform/sse_fanout.py`
- `src/polar/platform/api/{topology,tasks,sessions,submit,templates,events,models}.py`
- `src/polar/platform/templates/{calculator,swebench}.json`
- `src/polar/platform/templates/README.md`
- `tests/platform/test_*.py`
- `web/` (full Vite project; do not commit `node_modules` or `dist`)

Modified files:

- [src/polar/cli.py](../src/polar/cli.py) — register `serve_platform` subcommand.
- [src/polar/rollout/server.py](../src/polar/rollout/server.py) — add `GET /tasks`, `GET /tasks/{id}/sessions`, `GET /events`.
- [src/polar/rollout/manager.py](../src/polar/rollout/manager.py) — emit events into the pub/sub.
- [src/polar/rollout/pipeline.py](../src/polar/rollout/pipeline.py) — emit `session.state_changed` events at each stage transition.
- [src/polar/gateway/server.py](../src/polar/gateway/server.py) — add `GET /sessions`, `GET /sessions/{id}/completions`, `GET /events`.
- [src/polar/gateway/storage.py](../src/polar/gateway/storage.py) — add `CompletionWriter`, enqueue from `save_message`.
- [src/polar/config/topology.py](../src/polar/config/topology.py) — accept `gateway.completion_persistence` block; provide defaults.
- `pyproject.toml` — add deps: `watchdog`, `httpx[http2]` if not already pinned, and a build hook to ship `web/dist/` and `src/polar/platform/templates/*.json` in the wheel.

Untouched but reused:

- All schemas in `src/polar/{rollout,trajectory,agent,runtime}/models.py`.
- Submission scripts under `examples/`. They must continue to work unmodified.

---

## 13. Definition of done for v1

- All seven milestones merged.
- All manual checklist items in §9.3 pass on a fresh local environment following the `examples/calculator/README.md` setup.
- Documentation: a short README at `src/polar/platform/README.md` covering start command, ports, and a screenshot or two. Update top-level `README.md` with a one-liner pointing to the platform service.
- No regression in any existing test suite. No measurable proxy-latency regression with completion persistence enabled.
