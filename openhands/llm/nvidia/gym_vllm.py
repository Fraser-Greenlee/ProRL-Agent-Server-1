"""
Gym-compatible token-level generation using the Responses API (/v1/responses).

Talks to a Gym model server (e.g., LocalVLLMModel with return_token_id_information: true)
via direct HTTP POST, matching the interaction pattern shown in Gym/minimax_client_example.py.

The Gym model server handles internally:
  - Responses API → Chat Completions conversion
  - Reasoning parser (<think> tag extraction)
  - logprobs extraction → generation_token_ids
  - /tokenize call → prompt_token_ids
  - Token ID information packaging into response output items

This module handles:
  - Converting ProRL's Chat Completions-style messages → Responses API input format
  - Sending to /v1/responses via direct HTTP POST
  - Parsing the response output → same ModelResponse format as qwen3.request_response_tokens
"""

import json
import logging
import re
import uuid
from typing import Any

import requests as http_requests
from litellm import ModelResponse
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ── Message format conversion: ProRL Chat Completions → Responses API input ──


def _extract_text_content(content: Any) -> str:
    """Extract plain text from message content that may be a string or list of objects.

    Handles both formats:
      - str: "Hello world"
      - list: [{"type": "text", "text": "Hello world"}, ...]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text', '')
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return '\n'.join(parts) if parts else ''
    return str(content) if content else ''


def _convert_messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert ProRL's serialized Message dicts to Responses API input items.

    ProRL messages look like standard Chat Completions messages with extra fields
    (input_ids, output_ids, logprobs). The Responses API expects a flat list of
    typed items: message, reasoning, function_call, function_call_output.
    """
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg['role']
        content = _extract_text_content(msg.get('content', ''))

        if role in ('system', 'user'):
            input_items.append({
                'role': role,
                'content': content,
                'type': 'message',
            })

        elif role == 'assistant':
            _convert_assistant_message(msg, content, input_items)

        elif role == 'tool':
            input_items.append({
                'type': 'function_call_output',
                'call_id': msg.get('tool_call_id', ''),
                'output': _extract_text_content(msg.get('content', '')),
            })

    return input_items


def _convert_assistant_message(
    msg: dict[str, Any],
    content: str,
    input_items: list[dict[str, Any]],
) -> None:
    """Split a ProRL assistant message into Responses API items.

    An assistant message may contain:
      - <think>...</think> reasoning → reasoning item(s)
      - Remaining text content → message item
      - tool_calls → function_call item(s)
      - Token IDs (input_ids/output_ids/logprobs) → attached to last item
    """
    # Extract <think>...</think> reasoning blocks
    reasoning_matches = THINK_TAG_PATTERN.findall(content)
    remaining_content = THINK_TAG_PATTERN.sub('', content).strip()

    for reasoning_text in reasoning_matches:
        if reasoning_text.strip():
            input_items.append({
                'type': 'reasoning',
                'id': f'rs_{uuid.uuid4().hex}',
                'summary': [{'text': reasoning_text, 'type': 'summary_text'}],
                'status': 'completed',
            })

    last_item: dict[str, Any] | None = None

    # Add message item if there's content (or no tool calls at all)
    tool_calls = msg.get('tool_calls') or []
    if remaining_content or not tool_calls:
        msg_item: dict[str, Any] = {
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': remaining_content}],
            'type': 'message',
            'status': 'completed',
        }
        input_items.append(msg_item)
        last_item = msg_item

    # Add function_call items
    for tc in tool_calls:
        func = tc.get('function', tc)
        args = func.get('arguments', '')
        if isinstance(args, dict):
            args = json.dumps(args)
        call_id = tc.get('id', f'call_{uuid.uuid4().hex}')
        fc_item: dict[str, Any] = {
            'type': 'function_call',
            'name': func.get('name', ''),
            'arguments': args,
            'call_id': call_id,
            'id': call_id,
            'status': 'completed',
        }
        input_items.append(fc_item)
        last_item = fc_item

    # NOTE: Token IDs (input_ids, output_ids, logprobs) from previous responses
    # are NOT sent back to the Gym model server — they are output-only fields.
    # The Gym server re-tokenizes the full conversation on each request.
    # The token IDs are preserved in the rollout server's message history for training.


def _convert_tools_to_responses_format(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Chat Completions tool defs to Responses API tool format.

    Chat Completions: [{"type": "function", "function": {"name": ..., "parameters": ...}}]
    Responses API:    [{"type": "function", "name": ..., "parameters": ...}]
    """
    if not tools:
        return []
    result = []
    for tool in tools:
        if isinstance(tool, dict) and 'function' in tool:
            func = tool['function']
            result.append({
                'type': 'function',
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'parameters': func.get('parameters', {}),
                'strict': func.get('strict', False),
            })
        else:
            # Ensure 'strict' is present even for pre-formatted tools
            if isinstance(tool, dict) and 'strict' not in tool:
                tool = {**tool, 'strict': False}
            result.append(tool)
    return result


# ── Response parsing: Responses API output → ProRL internal format ──


def _parse_responses_output(
    output_items: list[dict[str, Any]],
) -> tuple[str, list[dict], list[int] | None, list[int] | None, list[float] | None]:
    """Parse Responses API output items into (content, tool_calls, prompt_ids, gen_ids, logprobs).

    The Gym model server returns output items in order:
      [reasoning, ...] [message] [function_call, ...]
    Token IDs are attached to the last item when return_token_id_information is true.
    """
    reasoning_parts: list[str] = []
    text_content = ''
    tool_calls: list[dict[str, Any]] = []
    prompt_token_ids: list[int] | None = None
    generation_token_ids: list[int] | None = None
    generation_log_probs: list[float] | None = None

    for item in output_items:
        item_type = item.get('type', '')

        if item_type == 'reasoning':
            for s in item.get('summary') or []:
                if s.get('type') == 'summary_text' and s.get('text'):
                    reasoning_parts.append(s['text'])

        elif item_type == 'message':
            for c in item.get('content') or []:
                if isinstance(c, dict) and c.get('type') == 'output_text':
                    text_content += c.get('text', '')

        elif item_type == 'function_call':
            args = item.get('arguments', '')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            tool_calls.append({
                'id': item.get('call_id', str(uuid.uuid4())),
                'type': 'function',
                'function': {
                    'name': item.get('name', ''),
                    'arguments': args,
                },
            })

        # Token IDs — can be on any item, typically the last one
        if 'generation_token_ids' in item:
            generation_token_ids = [int(x) for x in item['generation_token_ids']]
        if 'prompt_token_ids' in item:
            prompt_token_ids = [int(x) for x in item['prompt_token_ids']]
        if 'generation_log_probs' in item:
            generation_log_probs = [float(x) for x in item['generation_log_probs']]

    # Reconstruct full content with <think> tags (same as ProRL's internal format)
    content = ''
    if reasoning_parts:
        reasoning_text = ''.join(reasoning_parts)
        content = f'<think>{reasoning_text}</think>\n\n'
    content += text_content

    return content, tool_calls, prompt_token_ids, generation_token_ids, generation_log_probs


# ── Gym head server discovery ──

# Cache for the resolved model server URL
_resolved_base_url: str | None = None


def discover_gym_model_url(
    head_server_url: str = "http://127.0.0.1:11000",
    server_name: str | None = None,
) -> str:
    """Discover the Gym model server URL by querying the head server.

    The head server returns a global config YAML that contains the model
    server's host and port. This function parses that config and returns
    the base URL (http://host:port) for the first (or named) model server.

    Args:
        head_server_url: The Gym head server URL (default: http://127.0.0.1:11000)
        server_name: Optional server name to look up (default: use the first server)

    Returns:
        The model server base URL, e.g. "http://127.0.0.1:13883"
    """
    try:
        resp = http_requests.get(f"{head_server_url}/global_config_dict_yaml", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Gym head server at {head_server_url}. "
            f"Make sure the Gym server is running. Error: {e}"
        ) from e

    import yaml
    config = yaml.safe_load(json.loads(resp.text))

    if server_name and server_name in config:
        server_config = config[server_name]
    else:
        # Find the first server entry that has responses_api_models
        for key, value in config.items():
            if isinstance(value, dict) and 'responses_api_models' in value:
                server_config = value
                server_name = key
                break
        else:
            raise RuntimeError(
                f"No model server found in Gym config. Config keys: {list(config.keys())}"
            )

    # Traverse: server_name -> responses_api_models -> first model -> host/port
    models = server_config.get('responses_api_models', {})
    first_model = next(iter(models.values()))
    host = first_model.get('host', '127.0.0.1')
    port = first_model.get('port')
    if port is None:
        raise RuntimeError(f"No port found for model server {server_name}")

    url = f"http://{host}:{port}"
    logger.info(f"[gym_vllm] Discovered model server: {server_name} at {url}")
    return url


def _resolve_base_url(base_url: str) -> str:
    """Resolve the base URL for the Gym model server.

    If base_url points to the head server (port 11000), auto-discover the
    model server URL. Otherwise, use base_url directly.

    The resolved URL is cached globally so discovery only happens once.
    """
    global _resolved_base_url
    if _resolved_base_url is not None:
        return _resolved_base_url

    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    # Check if this looks like a head server URL (port 11000)
    if parsed.port == 11000:
        logger.info(f"[gym_vllm] base_url {base_url} appears to be head server, auto-discovering model server...")
        _resolved_base_url = discover_gym_model_url(
            head_server_url=f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        )
    else:
        # Strip /v1 suffix to get the base URL for building /v1/responses
        url = base_url.rstrip('/')
        if url.endswith('/v1'):
            _resolved_base_url = url[:-3]  # Remove /v1
        else:
            _resolved_base_url = url

    logger.info(f"[gym_vllm] Resolved base URL: {_resolved_base_url}")
    return _resolved_base_url


# ── Direct HTTP helper ──


def _build_responses_url(base_url: str) -> str:
    """Build the /v1/responses URL from the LLM base_url.

    Handles various base_url formats:
      - http://host:port/v1   → http://host:port/v1/responses
      - http://host:port/v1/  → http://host:port/v1/responses
      - http://host:port      → http://host:port/v1/responses
      - http://host:port/     → http://host:port/v1/responses
    """
    url = base_url.rstrip('/')
    if url.endswith('/v1'):
        return url + '/responses'
    else:
        return url + '/v1/responses'


def _post_responses(
    base_url: str,
    request_body: dict[str, Any],
    timeout: int | None = None,
) -> dict[str, Any]:
    """POST to /v1/responses via direct HTTP request (synchronous)."""
    url = _build_responses_url(base_url)
    logger.info(f"[gym_vllm] POST {url}")
    logger.debug(f"[gym_vllm] Request body keys: {list(request_body.keys())}")

    try:
        response = http_requests.post(
            url,
            json=request_body,
            timeout=timeout or 600,
            headers={'Content-Type': 'application/json'},
        )
    except http_requests.exceptions.ConnectionError as e:
        # Clear the cached URL so next call will re-discover
        global _resolved_base_url
        _resolved_base_url = None
        logger.error(f"[gym_vllm] Connection error to {url}: {e}")
        raise RuntimeError(
            f"Cannot connect to Gym model server at {url}. "
            f"Make sure the Gym server is running (bash scripts/support_gym/serve_gym_llm.sh). "
            f"Error: {e}"
        ) from e
    except http_requests.exceptions.Timeout as e:
        logger.error(f"[gym_vllm] Request timeout to {url}: {e}")
        raise RuntimeError(f"Request to {url} timed out after {timeout}s") from e

    if response.status_code >= 400:
        error_text = response.text[:2000]
        logger.error(
            f"[gym_vllm] POST {url} returned {response.status_code}: {error_text}"
        )
        # Log the full request body for debugging
        logger.error(
            f"[gym_vllm] Failed request had {len(request_body.get('input', []))} input items, "
            f"{len(request_body.get('tools', []))} tools"
        )
        # Log the types of each input item for debugging
        for i, item in enumerate(request_body.get('input', [])):
            item_type = item.get('type', 'unknown')
            item_role = item.get('role', 'N/A')
            content_type = type(item.get('content', '')).__name__
            logger.error(
                f"[gym_vllm]   input[{i}]: type={item_type}, role={item_role}, "
                f"content_type={content_type}, keys={list(item.keys())}"
            )
        raise RuntimeError(
            f"POST /v1/responses returned {response.status_code}: {error_text}"
        )

    resp_data = response.json()
    logger.info(
        f"[gym_vllm] Response received: status={resp_data.get('status', 'N/A')}, "
        f"output_items={len(resp_data.get('output', []))}"
    )
    return resp_data


# ── Main entry point ──


def request_response_tokens(
    tokenizer: AutoTokenizer,
    base_url: str | None,
    timeout: int | None,
    top_p: float,
    seed: int | None,
    max_model_len: int,
    api_key: str | None = None,
    server_name: str = "policy_model",
    **kwargs,
) -> ModelResponse:
    """Gym-compatible token-level generation via the Responses API (/v1/responses).

    Talks to a Gym model server (LocalVLLMModel / VLLMModel) configured with
    return_token_id_information: true via direct HTTP POST.
    The server handles reasoning parsing, logprobs extraction, and prompt
    tokenization internally.

    The returned ModelResponse has the same structure as qwen3.request_response_tokens:
      choices[0].input_ids   = prompt_token_ids
      choices[0].output_ids  = generation_token_ids
      choices[0].logprobs    = generation_log_probs
      choices[0].message     = {content, tool_calls, format_error_type}
    """
    messages = kwargs.pop('messages')
    model = kwargs.pop('model', None)
    tools = kwargs.pop('tools', [])
    chat_template_kwargs = kwargs.pop('chat_template_kwargs', {'enable_thinking': True})
    kwargs.pop('extra_body', None)

    max_output_tokens = kwargs.pop('max_completion_tokens', None) or kwargs.pop('max_tokens', None)
    temperature = kwargs.pop('temperature', None)

    logger.info(
        f"[gym_vllm] request_response_tokens called: base_url={base_url}, "
        f"model={model}, num_messages={len(messages)}, num_tools={len(tools)}, "
        f"max_output_tokens={max_output_tokens}, temperature={temperature}"
    )

    if not base_url:
        raise ValueError(
            "[gym_vllm] base_url is required for Gym API. "
            "Set it via the LLM server address (e.g. http://127.0.0.1:8000/v1)."
        )

    # ── 1) Convert ProRL messages → Responses API input ──
    input_items = _convert_messages_to_responses_input(messages)
    logger.debug(f"[gym_vllm] Converted {len(messages)} messages → {len(input_items)} input items")

    # ── 2) Convert tools format ──
    responses_tools = _convert_tools_to_responses_format(tools)

    # ── 3) Build request body (same shape as Responses API) ──
    request_body: dict[str, Any] = {
        'input': input_items,
    }
    if responses_tools:
        request_body['tools'] = responses_tools
    if max_output_tokens is not None:
        request_body['max_output_tokens'] = max_output_tokens
    if temperature is not None:
        request_body['temperature'] = temperature
    if top_p is not None:
        request_body['top_p'] = top_p

    # Pass chat_template_kwargs via metadata (how Gym's VLLMModel reads per-request overrides)
    if chat_template_kwargs:
        request_body.setdefault('metadata', {})
        request_body['metadata']['chat_template_kwargs'] = json.dumps(chat_template_kwargs)

    # ── 4) Resolve model server URL and POST /v1/responses ──
    resolved_url = _resolve_base_url(base_url)
    resp_data = _post_responses(resolved_url, request_body, timeout=timeout)

    # ── 5) Parse Responses API output ──
    output_items = resp_data.get('output') or []
    logger.debug(f"[gym_vllm] Parsing {len(output_items)} output items")
    (
        content,
        tool_calls,
        prompt_token_ids,
        generation_token_ids,
        generation_log_probs,
    ) = _parse_responses_output(output_items)

    logger.info(
        f"[gym_vllm] Parsed response: content_len={len(content)}, "
        f"tool_calls={len(tool_calls)}, "
        f"prompt_ids={'present' if prompt_token_ids else 'missing'}, "
        f"gen_ids={'present' if generation_token_ids else 'missing'}, "
        f"logprobs={'present' if generation_log_probs else 'missing'}"
    )

    # Determine format_error_type from incomplete_details
    incomplete = resp_data.get('incomplete_details')
    format_error_type = (
        'length'
        if incomplete and incomplete.get('reason') == 'max_output_tokens'
        else None
    )

    # ── 6) Build ModelResponse (same format as qwen3.request_response_tokens) ──
    random_id = f"chatcmpl-{uuid.uuid4()}"
    result = {
        'input_ids': prompt_token_ids or [],
        'output_ids': generation_token_ids or [],
        'logprobs': generation_log_probs or [],
        'message': {
            'content': content,
            'tool_calls': tool_calls,
            'format_error_type': format_error_type,
        },
    }

    model_response = ModelResponse(
        model=model or resp_data.get('model'),
        id=random_id,
        choices=[result],
        role='assistant',
    )
    logger.info(f"[gym_vllm] ModelResponse built: id={random_id}")
    return model_response
