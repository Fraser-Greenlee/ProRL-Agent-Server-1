# type: ignore
import time
import pandas as pd
import numpy as np
import os
import asyncio
import json
import copy

from evaluation.benchmarks.swe_bench.run_infer import (  # type: ignore
    initialize_runtime,
    get_instruction,
    complete_runtime,
    is_fatal_evaluation_error,
    codeact_user_response,
    EvalException
)
from evaluation.utils.shared import (  # type: ignore
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)
from evaluation.benchmarks.swe_bench.resource.mapping import get_instance_resource_factor

from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
)
from openhands.core.setup import create_agent
from openhands.core.main import create_runtime, run_controller
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
)

from openhands.nvidia.registry import JobDetails

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'


from evaluation.benchmarks.swe_bench.eval_infer import (  # type: ignore
    process_instance as _eval_process_instance,
    process_git_patch as _process_git_patch,
    ConditionalImports as _EvalConditionalImports,
    ConditionalImports as ConditionalImports,
)
from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import LLMConfig
import pandas as pd

try:
    from swegym.harness.grading import get_eval_report  # type: ignore
    from swegym.harness.run_evaluation import (
        APPLY_PATCH_FAIL,  # type: ignore
        APPLY_PATCH_PASS,  # type: ignore
    )
    from swegym.harness.test_spec import make_test_spec  # type: ignore
except ModuleNotFoundError:
    # Fall back to the regular SWE-Bench harness
    from swebench.harness.grading import get_eval_report  # type: ignore
    from swebench.harness.run_evaluation import (
        APPLY_PATCH_FAIL,  # type: ignore
        APPLY_PATCH_PASS,  # type: ignore
    )
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

def is_last_action_finish(state: State) -> bool:
    if state and state.history:
        last_action = next(
            (
                event
                for event in reversed(state.history)
                if isinstance(event, Action)
            ),
            None,
        )
        if isinstance(last_action, AgentFinishAction):
            return True
    return False

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    base_container_image = get_instance_docker_image(
        instance['instance_id']
    )
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.runtime_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset or "swebench",
        instance_id=instance['instance_id'],
    )

    # run as fakeroot
    # Currently set to False as some container require GLIBC_2.38
    sandbox_config.run_as_fakeroot = False

    """
    sandbox_config.runtime_container_image = (
        '/lustre/fsw/portfolios/nvr/users/mingjiel/root/singularity_images/'
        'xingyaoww_sweb.eval.x86_64.getmoto_s_moto-7365.sif'
    )
    """

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )

    # https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/agent_config.py

    # Think Tool
    # https://github.com/yidong72/OpenHands_internal/blob/5fe7578f45a847f39e6a22947b504f2167e98117/openhands/agenthub/codeact_agent/tools/think.py#L14
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=False, # not too sure what this does.
        enable_history_truncation=False, # turn off history truncation
        ensure_thinking_end_properly=True, # set to true. might need to be false for eval other models.
    )
    config.set_agent_config(agent_config)
    return config

async def initialize_agents(
        instance:pd.Series,
        llm_config: LLMConfig | None = None,
        sid:str | None = None,
        eval_output_dir:str = "/root",
        git_commit:str = "9f93e8a1532d6e1da4ea702f3dbd31d0f6b2fb3a",
        dataset:str = "swebench",
        data_split:str = "train",
        max_iterations:int = 1,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    # Fall back to a sensible default if the caller does not provide an
    # explicit ``llm_config`` (mirrors the behaviour of the old
    # ``initialize_agents`` implementation that lived in
    # ``scripts/test_local_agent.py``).

    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    metadata = EvalMetadata(
        agent_class="CodeActAgent",
        llm_config=llm_config,
        agent_config=None,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details={'mode': 'swe'},
        condenser_config=NoOpCondenserConfig(),
    )


    config = get_config(instance, metadata)

    metadata.details['runtime_failure_count'] = 0  # type: ignore[index]
    metadata.details['remote_runtime_resource_factor'] = (  # type: ignore[index]
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config, sid=sid)

    await runtime.connect()
    print(f"Runtime connected {runtime.sid}")

    try:
        initialize_runtime(runtime, instance, metadata)

    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    # Return the same triple expected by all current call-sites. The caller
    # already has easy access to *instance* so we no longer return it here
    # (this avoids the previous mismatch where some sites expected three
    # return values).
    return runtime, metadata, config

async def run_agent(
        runtime:Runtime,
        metadata:EvalMetadata,
        config:OpenHandsConfig,
        instance:pd.Series
    ) -> dict[str, object]:
    message_action = get_instruction(instance, metadata)
    try:
        agent = create_agent(config)
        state: State | None = await run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
            )

         # if fatal error, throw EvalError to trigger re-run
        if state is None:
            raise EvalException('Final state is None')
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise e

    # get messages from agent history
    try:
        initial_user_message = agent._get_initial_user_message(state.history) # type: ignore
        raw_messages = agent._get_messages(state.history, initial_user_message) # type: ignore
        messages = agent.llm.format_messages_for_llm(raw_messages)
        if messages[-1]['role'] != 'assistant':
            messages = messages[:-1]

        from openhands.llm.llm_utils import check_tools
        tools = check_tools(agent.tools, agent.llm.config)

        new_messages = []
        for message in messages:
            new_message = {'role': message['role']}
            if type(message['content']) == str:
                new_message['content'] = message['content']
            else:
                new_message['content'] = message['content'][0]['text']
            if 'tool_calls' in message:
                new_message['tool_calls'] = [tool_call['function'] for tool_call in message['tool_calls']]

            # Handle the case where the agent did not end properly reasoning properly
            # This largely formats the message to be consistent with the expected output from Qwen3 models.
            if message['role'] == 'assistant' and '<think>' in new_message['content'] and '</think>' not in new_message['content']:
                if 'tool_calls' in new_message:
                    tool_calls_message = ""
                    for tool_call in new_message['tool_calls']:
                        current_tool_call = []
                        current_tool_call.append('\n<tool_call>\n{"name": "' + tool_call['name'] + '", "arguments": ')
                        if isinstance(tool_call['arguments'], str):
                            current_tool_call.append(tool_call['arguments'])
                        else:
                            current_tool_call.append(json.dumps(tool_call['arguments']))
                        current_tool_call.append("}\n</tool_call>")
                        tool_calls_message += ''.join(current_tool_call)
                    new_message['content'] = f"{new_message['content']}\n{tool_calls_message}</think>"
                    new_message.pop('tool_calls')
                new_messages.append(new_message)
                return {
                    'git_patch': '',
                    'success': False,
                    'error': 'LLM did not end properly reasoning properly',
                    'finish': False,
                    'messages': new_messages,
                    'tools': tools
                }

            new_messages.append(new_message)
    except Exception as e:
        logger.error(f"Error while running, failed to retrieve agent messages: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    run_results = {
        "git_patch": git_patch,
        'success': not bool(state.last_error if state else True),
        'error': state.last_error if state and state.last_error else None,
        'finish': is_last_action_finish(state),
        'messages': new_messages,
        'tools': tools
    }
    return run_results

async def run(instance):
    #agent = initialize_agents(instance)
    #await initialize_runtime_for_agent(agent, instance)
    #return await run_agent(agent, instance)
    runtime, metadata, config = await initialize_agents(instance)
    results  = await run_agent(runtime, metadata, config, instance)
    return results


def _apply_patch_and_evaluate(runtime, git_patch: str, instance: pd.Series):

    from openhands.events.action import CmdRunAction
    from openhands.events.observation import CmdOutputObservation

    model_patch = _process_git_patch(git_patch)
    instance_id: str = instance["instance_id"]

    try:
        test_spec = make_test_spec(instance.to_dict())
    except Exception:
        from swebench.harness.utils import load_swebench_dataset

        dataset_name = "princeton-nlp/SWE-bench"
        full_dataset = load_swebench_dataset(dataset_name, "test")
        inst_map = {ins["instance_id"]: ins for ins in full_dataset}
        if instance_id not in inst_map:
            raise ValueError(f"Could not find instance_id {instance_id} in {dataset_name}.")
        test_spec = make_test_spec(inst_map[instance_id])

    import tempfile, os, time

    with tempfile.TemporaryDirectory() as tmp_dir:
        patch_path = os.path.join(tmp_dir, "patch.diff")
        with open(patch_path, "w") as f:
            f.write(model_patch)
        runtime.copy_to(patch_path, "/tmp")

        eval_script_path = os.path.join(tmp_dir, "eval.sh")
        with open(eval_script_path, "w") as f:
            f.write(test_spec.eval_script)
        runtime.copy_to(eval_script_path, "/tmp")

    # Make script executable
    action = CmdRunAction(command="chmod +x /tmp/eval.sh")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    apply_cmd = (
        "cd /testbed && "
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    patch_result = obs.content  # type: ignore[attr-defined]

    if "APPLY_PATCH_FAIL" in patch_result:
        raise RuntimeError(f"{instance_id}: {APPLY_PATCH_FAIL}\n{patch_result}")
    if "APPLY_PATCH_PASS" not in patch_result:
        raise RuntimeError(
            f"{instance_id}: Unexpected output when applying patch:\n{patch_result}"
        )

    logger.info(f"[{instance_id}] {APPLY_PATCH_PASS}:\n{patch_result}")

    log_file = "/tmp/eval_output.log"
    action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")
    action.set_hard_timeout(300)
    obs = runtime.run_action(action)
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError("Failed to launch evaluation script")

    pid = obs.content.split()[-1].strip()
    logger.info(f"[{instance_id}] Evaluation started (PID={pid})")

    start_time = time.time()
    timeout = 20 * 60  # 20 minutes
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
        check_action = CmdRunAction(command=f"ps -p {pid} > /dev/null; echo $?")
        check_action.set_hard_timeout(300)
        check_obs = runtime.run_action(check_action)
        if (
            isinstance(check_obs, CmdOutputObservation)
            and check_obs.content.split()[-1].strip() == "1"
        ):
            logger.info(f"[{instance_id}] Evaluation finished after {elapsed:.0f}s")
            break
        logger.info(f"[{instance_id}] [{elapsed:.0f}s] Evaluation in progress …")
        time.sleep(30)

    cat_action = CmdRunAction(command=f"cat {log_file}")
    cat_action.set_hard_timeout(300)
    cat_obs = runtime.run_action(cat_action)
    if not (isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0):
        raise RuntimeError("Failed to read evaluation output")

    test_output: str = cat_obs.content  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = os.path.join(tmp_dir, "logs", instance_id.lower())
        os.makedirs(logs_dir, exist_ok=True)
        test_out_path = os.path.join(logs_dir, "test_output.txt")
        with open(test_out_path, "w") as f:
            f.write(test_output)

        grading_report = get_eval_report(
            test_spec=test_spec,
            prediction={"model_patch": model_patch, "instance_id": instance_id},
            log_path=test_out_path,
            include_tests_status=True,
        )

    report = grading_report[instance_id]
    logger.info(f"[{instance_id}] Grading report: {report}")

    test_result = {
        "report": {
            "empty_generation": False,
            "resolved": report.get("resolved", False),
            "failed_apply_patch": "APPLY_PATCH_FAIL" in patch_result,
            "error_eval": False,
            "test_timeout": False,
        },
        "apply_patch_output": patch_result,
        "test_output": test_output,
    }

    return test_result

async def evaluate_agent(git_patch: str | None, instance: pd.Series, sid: str | None = None, allow_skip=True):
    # skip evaluation if git_patch is None or empty
    if allow_skip:
        if git_patch is None or len(git_patch) == 0:
            return {'resolved': False}
    else:
        if git_patch is None:
            raise ValueError('Patch is None, cannot evaluate')

    from openhands.utils.async_utils import call_sync_from_async

    runtime = None

    # Create a dummy LLM config to avoid the error of no LLM config
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        modify_params=False,
        log_completions=False,
    )

    try:
        runtime, _, _ = await initialize_agents(instance, llm_config=llm_config, sid=sid)
        test_result = await call_sync_from_async(
            _apply_patch_and_evaluate, runtime, git_patch, instance
        )
    except Exception as e:
        logger.error(f"Error evaluating agent: {e}")
        raise e
    finally:
        if runtime is not None:
            try:
                runtime.event_stream.close()
            except Exception:
                pass
            await call_sync_from_async(runtime.close)

    return test_result

def initialize_exception(job_details: JobDetails, e: Exception):
    instance_id = job_details.instance.instance_id if job_details.instance is not None else None
    trajectory_id = job_details.instance.trajectory_id if job_details.instance is not None else None
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in init: {str(e)}',
        'finish': False,
        'messages': [],
        'resolved': False,
        'critical_error': 'init',
    }

def run_exception(job_details: JobDetails, e: Exception):
    instance_id = job_details.instance.instance_id if job_details.instance is not None else None
    trajectory_id = job_details.instance.trajectory_id if job_details.instance is not None else None
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in run agent: {str(e)}',
        'finish': False,
        'messages': [],
        'resolved': False,
        'critical_error': 'run',
    }

def eval_exception(job_details: JobDetails, e: Exception):
    instance_id = job_details.instance.instance_id if job_details.instance is not None else None
    trajectory_id = job_details.instance.trajectory_id if job_details.instance is not None else None
    git_patch = job_details.run_results.get('git_patch', None) if job_details.run_results is not None else None
    success = job_details.run_results.get('success', False) if job_details.run_results is not None else False
    finish = job_details.run_results.get('finish', False) if job_details.run_results is not None else False
    messages = job_details.run_results.get('messages', []) if job_details.run_results is not None else []
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': git_patch,
        'success': success,
        'error': f'Error in eval: {str(e)}',
        'finish': finish,
        'messages': messages,
        'resolved': False,
        'critical_error': 'eval',
    }

def final_result(job_details: JobDetails):
    if job_details.results is None:
        if job_details.run_results is None:
            instance_id = job_details.instance.instance_id if job_details.instance is not None else None
            trajectory_id = job_details.instance.trajectory_id if job_details.instance is not None else None
            result = {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'resolved': job_details.eval_results['resolved'] if job_details.eval_results is not None else False,
                'critical_error': None,
            }
        else:
            result = {
                **job_details.run_results,
                'resolved': job_details.eval_results['resolved'] if job_details.eval_results is not None else False,
                'critical_error': None,
            }
    else:
        result = copy.deepcopy(job_details.results)
    if job_details.start_run_time:
        if job_details.start_time is None:
            job_details.start_time = job_details.start_run_time
        if job_details.end_time is None:
            job_details.end_time = job_details.start_run_time
        init_time_taken = job_details.start_run_time - job_details.start_time
        if job_details.start_eval_time:
            run_time_taken = (
                job_details.start_eval_time - job_details.start_run_time
            )
            evaluate_time_taken = job_details.end_time - job_details.start_eval_time
        else:
            run_time_taken = job_details.end_time - job_details.start_run_time
            evaluate_time_taken = 0
    else:
        if job_details.start_time is None or job_details.end_time is None:
            init_time_taken = 0
        else:
            init_time_taken = job_details.end_time - job_details.start_time
        run_time_taken = 0
        evaluate_time_taken = 0

    return {
            **result,
            'init_time_taken': init_time_taken,
            'run_time_taken': run_time_taken,
            'evaluate_time_taken': evaluate_time_taken,
        }
