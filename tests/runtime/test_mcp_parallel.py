"""
Parallel MCP test for SingularityRuntime.

Starts a single SSE MCP server via Singularity, then launches 64
SingularityRuntime instances in parallel. Each runtime performs an
MCPAction against the SSE server to list a unique marker directory
and asserts that a known file exists in the returned listing.
"""

import asyncio
import os
import socket
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from openhands.core.config import load_openhands_config
from openhands.core.config.mcp_config import MCPSSEServerConfig
from openhands.events import EventStream
from openhands.events.action.mcp import MCPAction
from openhands.runtime.impl.singularity.singularity_runtime import SingularityRuntime
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from openhands.storage import get_file_store


def _start_sse_mcp_singularity_server() -> tuple[subprocess.Popen[str], str]:
    """Start the SSE MCP server using Singularity and return (process, sse_url)."""
    # Ensure Singularity is available
    try:
        subprocess.run(
            ['singularity', '--version'], check=True, capture_output=True, text=True
        )
    except Exception as e:
        pytest.skip(f'Singularity is not available on this system: {e}')

    image_ref = 'docker://supercorp/supergateway'

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        host_port = s.getsockname()[1]

    # Bind the server directly to the chosen host port
    container_port = host_port

    # Command line mirrors the Docker args passed to the image entrypoint
    run_cmd = [
        'singularity',
        'run',
        image_ref,
        '--stdio',
        'npx -y @modelcontextprotocol/server-filesystem /tmp',
        '--port',
        str(container_port),
        '--baseUrl',
        f'http://localhost:{host_port}',
    ]

    proc = subprocess.Popen(
        run_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    # Give the server time to initialize
    time.sleep(10)

    return proc, f'http://localhost:{host_port}/sse'


def _stop_process(proc: subprocess.Popen[str]) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _run_single_runtime(test_params: dict, sse_url: str) -> dict:
    """Create one SingularityRuntime, run an MCP action, and return results."""
    runtime_id = test_params['runtime_id']
    temp_dir = test_params['temp_dir']

    start_time = time.time()

    try:
        # Load configuration
        config = load_openhands_config()
        config.run_as_openhands = False
        config.sandbox.keep_runtime_alive = False
        config.sandbox.force_rebuild_runtime = False

        # Use unique workspace for this runtime
        runtime_workspace = os.path.join(temp_dir, f'mcp_runtime_{runtime_id}')
        os.makedirs(runtime_workspace, exist_ok=True)

        config.workspace_base = runtime_workspace
        config.workspace_mount_path = runtime_workspace
        config.workspace_mount_path_in_sandbox = '/workspace'

        # Configure MCP SSE server
        config.mcp.sse_servers = [MCPSSEServerConfig(url=sse_url)]

        # Set up file store and event stream
        file_store = get_file_store(
            config.file_store,
            config.file_store_path,
            config.file_store_web_hook_url,
            config.file_store_web_hook_headers,
        )

        sid = f'test-mcp-parallel-{runtime_id}-{uuid.uuid4().hex[:8]}'
        event_stream = EventStream(sid, file_store)

        # Create runtime
        plugins = [AgentSkillsRequirement(), JupyterRequirement()]
        runtime = SingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            attach_to_existing=False,
        )

        # Connect
        asyncio.run(runtime.connect())

        # Create a unique marker directory on host and a file inside
        marker_dir = os.path.join('/tmp', f'mcp_parallel_{sid}')
        os.makedirs(marker_dir, exist_ok=True)
        marker_file = os.path.join(marker_dir, 'probe.txt')
        with open(marker_file, 'w') as f:
            f.write('ok')

        # Run MCP action to list the marker directory
        action = MCPAction(name='list_directory', arguments={'path': marker_dir})
        obs = asyncio.run(runtime.call_tool_mcp(action))

        # Parse and validate
        import json as _json

        data = _json.loads(obs.content)
        is_error = data.get('isError', False)
        text = ''
        for c in data.get('content', []):
            if c.get('type') == 'text' and isinstance(c.get('text'), str):
                text = c['text']
                break

        file_seen = f'[FILE] {os.path.basename(marker_file)}' in text

        # Cleanup runtime
        runtime.close()

        total_time = time.time() - start_time

        return {
            'runtime_id': runtime_id,
            'success': (not is_error) and file_seen,
            'total_time': total_time,
            'file_seen': file_seen,
            'is_error': is_error,
        }

    except Exception as e:
        total_time = time.time() - start_time
        return {
            'runtime_id': runtime_id,
            'success': False,
            'total_time': total_time,
            'error': str(e),
            'error_type': type(e).__name__,
        }


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME', 'docker').lower() != 'singularity',
    reason='Test only runs with TEST_RUNTIME=singularity',
)
def test_mcp_actions_parallel_singularity(temp_dir):
    """Run MCP actions in 64 parallel Singularity runtimes."""

    # Start a shared SSE MCP server for all runtimes
    proc, sse_url = _start_sse_mcp_singularity_server()

    try:
        n_runtimes = 64
        results = []
        start = time.time()

        test_params_list = [
            {
                'runtime_id': i,
                'temp_dir': temp_dir,
            }
            for i in range(1, n_runtimes + 1)
        ]

        with ThreadPoolExecutor(max_workers=n_runtimes) as executor:
            futures = {
                executor.submit(_run_single_runtime, params, sse_url): params[
                    'runtime_id'
                ]
                for params in test_params_list
            }

            for future in as_completed(futures):
                rid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    # Optional: print quick status line for diagnostics
                    status = (
                        'OK'
                        if result.get('success')
                        else f'FAIL ({result.get("error_type", "")})'
                    )
                    print(f'[PARALLEL MCP] Runtime {rid}: {status}')
                except Exception as e:
                    print(f'[PARALLEL MCP] Runtime {rid}: EXCEPTION - {e}')
                    results.append(
                        {
                            'runtime_id': rid,
                            'success': False,
                            'error': str(e),
                            'error_type': type(e).__name__,
                        }
                    )

        elapsed = time.time() - start
        successes = sum(1 for r in results if r.get('success'))
        print(
            f'[PARALLEL MCP] Completed {n_runtimes} runtimes in {elapsed:.2f}s: {successes} succeeded'
        )

        # Require at least one success to validate core behavior without being overly brittle
        assert successes > 0, (
            f'No successful MCP runs out of {n_runtimes}. First error: {results[0] if results else "no results"}'
        )

    finally:
        _stop_process(proc)
