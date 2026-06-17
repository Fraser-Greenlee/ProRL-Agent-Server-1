# LiteResearcher (Notre / Hermes)

Rollout-only example: run the shared [`notre_hermes`](../../src/polar/agent/presets/notre_hermes.py)
agent over the [LiteResearcher-Data](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data)
deep-research questions and grade the final `<answer>` with the
[`answer_judge`](../../src/polar/trajectory/evaluator/answer_judge.py) evaluator.

- **Dataset:** `simplex-ai-inc/LiteResearcher-Data` (`stage1`/`stage2`) — research
  questions with reference answers.
- **Environment:** [LiteResearcher](https://github.com/simplexai-labs/LiteResearcher).
  A `search`/`visit` MCP server ([`literesearcher_mcp_server.py`](literesearcher_mcp_server.py))
  proxies to the project's **live retrieval service** (Milvus + BGE-M3 hybrid
  search over the LiteResearcher corpus).
- **Agent:** `notre_hermes`. The MCP `search`/`visit` tools provide the research
  loop; `mcp` + `httpx` are baked into the runtime image.

## 1. Stand up the LiteResearcher retrieval service

Follow the upstream
[Environment README](https://github.com/simplexai-labs/LiteResearcher) to build
the Milvus index from the
[LiteResearcher-Corpus](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus)
and start the server (defaults to port `8018`):

```bash
git clone https://github.com/simplexai-labs/LiteResearcher.git
cd LiteResearcher/Environment
# ... build the Milvus index (see their README) ...
cd server && REDIS_HOST=127.0.0.1 EMBED_WORKERS=1 bash start.sh   # serves :8018
```

The MCP server calls `POST /search` and `POST /web_parser`. `/web_parser`
(full-document fetch) needs the server's optional PostgreSQL full-text storage
(`ENABLE_SQL_FULLTEXT=true`); without it, `visit` degrades to search snippets.
The service must be reachable from the runtime containers — with `network: host`
the default `http://127.0.0.1:8018` works when it runs on the rollout host.

## 2. Build the runtime image and cache the dataset

```bash
uv run python examples/lite_researcher_notre/build_image.py
```

The dataset is downloaded and cached on first submit.

## 3. Serve the model and start Polar

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3.6-27B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3

polar serve_rollout -c examples/lite_researcher_notre/topology.vllm.yaml
polar serve_gateway -c examples/lite_researcher_notre/topology.vllm.yaml --node-id localhost-node-01
polar serve_gateway -c examples/lite_researcher_notre/topology.vllm.yaml --node-id localhost-node-02
```

## 4. Submit rollouts

```bash
uv run python examples/lite_researcher_notre/submit_lite_tasks.py --max-tasks 10
uv run python examples/lite_researcher_notre/submit_lite_tasks.py --stage stage2 --max-tasks 50 --num-samples 4
uv run python examples/lite_researcher_notre/submit_lite_tasks.py \
  --search-url http://127.0.0.1:8018/search --parser-url http://127.0.0.1:8018/web_parser
```

Pass `--max-tasks -1` to run the full stage. Watch live progress with
`polar dashboard -c examples/lite_researcher_notre/topology.vllm.yaml`.
