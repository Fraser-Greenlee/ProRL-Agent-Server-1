# OpenHands NVIDIA LLM Module

This module provides NVIDIA-specific LLM implementations and utilities for the OpenHands framework, with specialized support for Qwen3 models and custom chat template handling.

## Overview

The NVIDIA LLM module is designed to interface with NVIDIA's hosted VLLM servers and provides optimized implementations for specific model families, particularly Qwen3 models with enhanced thinking capabilities and tool calling support.

## Key Features

- **Custom Qwen3 Chat Template**: Modified chat template that preserves thinking content in the content field
- **Token-based Communication**: Direct token-level communication with VLLM servers for improved performance
- **Tool Calling Support**: Native support for function calling with XML-based tool call formatting
- **Thinking Mode Support**: Enhanced support for models with reasoning/thinking capabilities
- **Context Window Management**: Automatic validation of input length against model context limits

## Components

### 1. Core Functions

#### `convert_messages_to_tokens()`
Converts OpenAI-format chat messages to token sequences for VLLM processing.

**Parameters:**
- `messages`: List of chat messages in OpenAI format
- `tokenizer`: HuggingFace tokenizer instance
- `chat_template`: Optional custom chat template (defaults to Qwen3 template)
- `add_generation_prompt`: Whether to add generation prompt for the last message
- `enable_thinking`: Enable thinking mode for reasoning models
- `tools`: List of available tools for function calling

**Returns:** List of token sequences, one for each message

#### `parse_response_ids()`
Parses token IDs from VLLM response into structured message format with tool calls.

**Parameters:**
- `response_ids`: List of token IDs from VLLM response
- `tokenizer`: HuggingFace tokenizer for decoding

**Returns:** Dictionary with `content` and `tool_calls` fields

#### `request_response_tokens()`
Main interface function that handles the complete request lifecycle to VLLM servers.

**Parameters:**
- `base_url`: VLLM server endpoint
- `tokenizer`: Model tokenizer
- `max_model_len`: Maximum context length for the model
- `messages`: Chat conversation history
- `top_p`: Nucleus sampling parameter
- `seed`: Random seed for reproducibility
- `timeout`: Request timeout
- `**kwargs`: Additional parameters (max_tokens, temperature, etc.)

**Returns:** LiteLLM ModelResponse object

### 2. Custom Chat Template

The module includes a specialized chat template (`qwen3_chat_template`) that addresses specific issues with Qwen3 models:

- **Thinking Content Preservation**: Ensures thinking content remains in the content field under all conditions
- **Tool Integration**: Seamless integration of tool definitions and responses
- **Multi-step Tool Support**: Proper handling of multi-step tool calling scenarios

## Usage Examples

### Basic Text Completion

```python
from transformers import AutoTokenizer
from openhands.llm.nvidia.qwen3 import request_response_tokens

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

response = request_response_tokens(
    base_url="http://your-vllm-server:8000",
    tokenizer=tokenizer,
    max_model_len=32768,
    messages=messages,
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0]['message']['content'])
```

### Tool Calling

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather in New York?"}
]

response = request_response_tokens(
    base_url="http://your-vllm-server:8000",
    tokenizer=tokenizer,
    max_model_len=32768,
    messages=messages,
    tools=tools,
    max_tokens=1024
)

# Check for tool calls
if response.choices[0]['message']['tool_calls']:
    for tool_call in response.choices[0]['message']['tool_calls']:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

### Thinking Mode

```python
# Enable thinking mode for reasoning tasks
response = request_response_tokens(
    base_url="http://your-vllm-server:8000",
    tokenizer=tokenizer,
    max_model_len=32768,
    messages=messages,
    chat_template_kwargs={'enable_thinking': True},
    max_tokens=2048
)
```

## Integration with OpenHands

This module integrates with the broader OpenHands LLM infrastructure:

- **LLM Base Classes**: Compatible with `LLM`, `AsyncLLM`, and `StreamingLLM` interfaces
- **Configuration**: Uses OpenHands LLM configuration system
- **Logging**: Integrated with OpenHands logging infrastructure
- **Error Handling**: Consistent error handling with other LLM providers

## Model Support

Currently optimized for:
- **Qwen3 Series**: With enhanced thinking capabilities
- **Tool-enabled Models**: Models trained for function calling

## Requirements

- `transformers`: For tokenizer support
- `httpx`: For HTTP requests to VLLM servers
- `litellm`: For response formatting
- `pydantic`: For configuration management

## Error Handling

The module includes robust error handling for:
- **Context Window Exceeded**: Automatic detection and reporting
- **Server Communication**: HTTP error handling with proper status codes
- **Response Parsing**: Graceful handling of malformed responses
- **Token Conversion**: Validation of token sequences

## Performance Considerations

- **Token-level Communication**: Bypasses text encoding/decoding overhead
- **Batch Processing**: Efficient handling of multiple messages
- **Context Validation**: Pre-request validation to avoid wasted calls
- **Connection Reuse**: HTTP session management for reduced latency
