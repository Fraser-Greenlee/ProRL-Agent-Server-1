"""
Debug test for SingularityRuntime parallel execution issues.

This test is designed to gather detailed diagnostics about why
SingularityRuntime instances fail to initialize in parallel.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from openhands.core.config import load_openhands_config
from openhands.events import EventStream
from openhands.events.action.commands import CmdRunAction
from openhands.runtime.impl.singularity.singularity_runtime import SingularityRuntime
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from openhands.storage import get_file_store


def create_and_test_runtime(test_params):
    """Create a single SingularityRuntime and test it.

    Args:
        test_params: Dict with 'runtime_id', 'temp_dir', 'delay'

    Returns:
        Dict with test results
    """
    runtime_id = test_params['runtime_id']
    temp_dir = test_params['temp_dir']
    delay = test_params.get('delay', 0)

    print(f'\n[RUNTIME {runtime_id}] === STARTING DEBUG PROCESS ===')

    if delay > 0:
        print(f'[RUNTIME {runtime_id}] Delaying start by {delay} seconds...')
        time.sleep(delay)

    start_time = time.time()

    try:
        # Load configuration
        config = load_openhands_config()
        config.run_as_openhands = False
        config.sandbox.keep_runtime_alive = False
        config.sandbox.force_rebuild_runtime = False

        # Use unique workspace for this runtime
        runtime_workspace = os.path.join(temp_dir, f'runtime_{runtime_id}')
        os.makedirs(runtime_workspace, exist_ok=True)

        config.workspace_base = runtime_workspace
        config.workspace_mount_path = runtime_workspace
        config.workspace_mount_path_in_sandbox = '/workspace'

        print(f'[RUNTIME {runtime_id}] Workspace: {runtime_workspace}')

        # Set up file store and event stream
        file_store = get_file_store(
            config.file_store,
            config.file_store_path,
            config.file_store_web_hook_url,
            config.file_store_web_hook_headers,
        )

        sid = f'test-runtime-{runtime_id}'
        event_stream = EventStream(sid, file_store)

        print(f'[RUNTIME {runtime_id}] Creating SingularityRuntime with SID: {sid}')

        # Create runtime
        plugins = [AgentSkillsRequirement(), JupyterRequirement()]
        runtime = SingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            attach_to_existing=False,
        )

        print(f'[RUNTIME {runtime_id}] Attempting to connect...')
        connection_start = time.time()

        # Try to connect
        asyncio.run(runtime.connect())

        connection_time = time.time() - connection_start
        print(
            f'[RUNTIME {runtime_id}] Connected successfully in {connection_time:.2f}s'
        )

        # Test with a simple command
        print(f'[RUNTIME {runtime_id}] Testing with echo command...')
        action = CmdRunAction(command=f'echo "hello world {runtime_id}"')
        obs = runtime.run_action(action)

        print(
            f'[RUNTIME {runtime_id}] Command result: exit_code={obs.exit_code}, output="{obs.content.strip()}"'
        )

        # Clean up
        runtime.close()

        total_time = time.time() - start_time

        return {
            'runtime_id': runtime_id,
            'success': True,
            'total_time': total_time,
            'connection_time': connection_time,
            'command_output': obs.content.strip(),
            'exit_code': obs.exit_code,
            'api_url': runtime.api_url,
            'container_port': runtime._container_port,
            'container_pid': runtime.container_pid,
        }

    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)

        print(f'[RUNTIME {runtime_id}] FAILED after {total_time:.2f}s: {error_msg}')

        return {
            'runtime_id': runtime_id,
            'success': False,
            'total_time': total_time,
            'error': error_msg,
            'error_type': type(e).__name__,
        }


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME', 'docker').lower() != 'singularity',
    reason='Test only runs with TEST_RUNTIME=singularity',
)
def test_singularity_runtime_parallel_debug(temp_dir):
    """Debug test for SingularityRuntime parallel execution."""

    print('\n=== DEBUGGING SINGULARITY RUNTIME PARALLEL EXECUTION ===')
    print(f'Workspace: {temp_dir}')

    n_runtimes = 4

    # Test 1: Sequential execution (baseline)
    print(f'\n--- TEST 1: Sequential execution of {n_runtimes} runtimes ---')
    sequential_results = []
    sequential_start = time.time()

    for i in range(1, n_runtimes + 1):
        test_params = {
            'runtime_id': i,
            'temp_dir': temp_dir,
            'delay': 0,
        }
        result = create_and_test_runtime(test_params)
        sequential_results.append(result)

        if result['success']:
            print(f'[SEQUENTIAL] Runtime {i}: SUCCESS in {result["total_time"]:.2f}s')
        else:
            print(
                f'[SEQUENTIAL] Runtime {i}: FAILED in {result["total_time"]:.2f}s - {result["error"]}'
            )

    sequential_time = time.time() - sequential_start
    print(f'[SEQUENTIAL] Total time: {sequential_time:.2f}s')

    # Test 2: Parallel execution (debug focus)
    print(f'\n--- TEST 2: Parallel execution of {n_runtimes} runtimes ---')
    parallel_results = []
    parallel_start = time.time()

    test_params_list = [
        {
            'runtime_id': i,
            'temp_dir': temp_dir,
            'delay': 0,  # No staggered delays initially
        }
        for i in range(1, n_runtimes + 1)
    ]

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=n_runtimes) as executor:
        # Submit all tasks
        future_to_runtime = {
            executor.submit(create_and_test_runtime, params): params['runtime_id']
            for params in test_params_list
        }

        # Collect results as they complete
        for future in as_completed(future_to_runtime):
            runtime_id = future_to_runtime[future]
            try:
                result = future.result()
                parallel_results.append(result)

                if result['success']:
                    print(
                        f'[PARALLEL] Runtime {runtime_id}: SUCCESS in {result["total_time"]:.2f}s'
                    )
                else:
                    print(
                        f'[PARALLEL] Runtime {runtime_id}: FAILED in {result["total_time"]:.2f}s - {result["error"]}'
                    )

            except Exception as e:
                print(f'[PARALLEL] Runtime {runtime_id}: EXCEPTION - {e}')
                parallel_results.append(
                    {
                        'runtime_id': runtime_id,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                    }
                )

    parallel_time = time.time() - parallel_start
    print(f'[PARALLEL] Total time: {parallel_time:.2f}s')

    # Analyze results
    print('\n--- ANALYSIS ---')

    sequential_successes = sum(1 for r in sequential_results if r['success'])
    parallel_successes = sum(1 for r in parallel_results if r['success'])

    print(f'Sequential: {sequential_successes}/{n_runtimes} successful')
    print(f'Parallel: {parallel_successes}/{n_runtimes} successful')

    if sequential_successes > 0 and parallel_successes == 0:
        print(
            '\n*** ISSUE CONFIRMED: Runtimes work sequentially but fail in parallel ***'
        )

        # Show error details
        print('\nParallel execution errors:')
        for result in parallel_results:
            if not result['success']:
                print(
                    f'  Runtime {result["runtime_id"]}: {result["error_type"]} - {result["error"]}'
                )

        # This would normally fail the test, but for debugging we want to see the output
        print('\n*** DIAGNOSTIC TEST COMPLETE - CHECK LOGS FOR DETAILED ERROR INFO ***')

    elif parallel_successes == n_runtimes:
        print('\n*** SUCCESS: Parallel execution works! ***')
        efficiency = sequential_time / parallel_time if parallel_time > 0 else 1.0
        print(f'Efficiency ratio: {efficiency:.2f}x')

    else:
        print(
            f'\n*** PARTIAL SUCCESS: {parallel_successes}/{n_runtimes} runtimes succeeded in parallel ***'
        )

    # For debugging purposes, let's not fail the test, just report findings
    print('\n=== DEBUG TEST COMPLETED ===')


if __name__ == '__main__':
    # For manual testing
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_singularity_runtime_parallel_debug(temp_dir)
