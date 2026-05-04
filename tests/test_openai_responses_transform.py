"""Tests for OpenAI Responses API request conversion."""

from __future__ import annotations

from polar.gateway.transform.openai_responses import OpenAIResponsesTransformer


def test_responses_flat_function_tools_are_forwarded_to_chat_tools() -> None:
    body = {
        "instructions": "system",
        "input": "hello",
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Runs a command.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            }
        ],
    }

    converted = OpenAIResponsesTransformer().transform_request(body)
    function = converted["tools"][0]["function"]

    assert converted["tool_choice"] == "auto"
    assert function["name"] == "exec_command"
    assert sorted(function["parameters"]["properties"]) == ["cmd", "workdir"]
    assert not any("tools" in message for message in converted["messages"])


def test_responses_input_schema_variants_are_preserved() -> None:
    body = {
        "input": "hello",
        "tools": [
            {
                "type": "function",
                "name": "legacy_camel",
                "inputSchema": {
                    "jsonSchema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                    }
                },
            },
            {
                "type": "function",
                "name": "legacy_snake",
                "input_schema": {
                    "type": "object",
                    "properties": {"session_id": {"type": "integer"}},
                },
            },
        ],
    }

    converted = OpenAIResponsesTransformer().transform_request(body)
    tools = {tool["function"]["name"]: tool["function"] for tool in converted["tools"]}

    assert sorted(tools["legacy_camel"]["parameters"]["properties"]) == ["cmd"]
    assert sorted(tools["legacy_snake"]["parameters"]["properties"]) == ["session_id"]
