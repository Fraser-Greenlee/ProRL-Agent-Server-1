# type: ignore
import time
import pandas as pd
import numpy as np
import os
import asyncio
import json
import copy
import traceback
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
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
)

from openhands.nvidia.registry import JobDetails
from openhands.nvidia.utils import process_messages_from_agent_state

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

from swegym.harness.grading import get_eval_report
from swegym.harness.run_evaluation import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)
from swegym.harness.test_spec import make_test_spec

def get_instance_docker_image(instance_id: str, dataset: str | None = None) -> str:
    is_multimodal = bool(dataset and 'multimodal' in dataset.lower())
    print("is_multimodal", is_multimodal)
    if is_multimodal:
        try:
            repo, issue = instance_id.split('__', 1)
        except ValueError:
            repo, issue = instance_id, ''

        _issue = issue[:-2] if issue.endswith("_0") else issue
        image_name = f"sweb.eval.x86_64.{repo}_1776_{_issue}:latest"
        docker_prefix = "docker.io/swebench"
    else:
        image_name = f"sweb.eval.x86_64.{instance_id}".replace("__", "_s_")
        docker_prefix = DOCKER_IMAGE_PREFIX.rstrip("/")
    return f"{docker_prefix}/{image_name}".lower()

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
    instance: dict,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    if 'image_assets' in instance.keys():
        is_multimodal = True
    else:
        is_multimodal = False
    base_container_image = get_instance_docker_image(
        instance['instance_id'], "swebench" if not is_multimodal else "swebench_multimodal"
    )
    logger.debug(
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
        instance: dict,
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
    if 'image_assets' in instance.keys():
        dataset = "swebench_multimodal"

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
    logger.debug(f"Runtime connected {runtime.sid}")

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
        instance:dict,
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
        logger.debug(
            f'Got git diff for instance {instance['instance_id']}:\n--------\n{git_patch}\n--------'
        )

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise e

    # get messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state) # type: ignore
    except Exception as e:
        logger.error(f"Error while running, failed to retrieve agent messages: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    return {
        "git_patch": git_patch if run_results['end_properly'] else "",
        'success': not bool(state.last_error if state else True),
        'error': state.last_error if state and state.last_error else None,
        'finish': is_last_action_finish(state),
        **run_results,
    }

async def run(instance):
    #agent = initialize_agents(instance)
    #await initialize_runtime_for_agent(agent, instance)
    #return await run_agent(agent, instance)
    runtime, metadata, config = await initialize_agents(instance)
    results  = await run_agent(runtime, metadata, config, instance)
    return results


def _apply_patch_and_evaluate(
    runtime,
    git_patch: str,
    instance: dict,
):

    if 'image_assets' in instance.keys():
        is_multimodal = True
    else:
        is_multimodal = False

    if is_multimodal:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swebench.harness.test_spec.test_spec import make_test_spec
    else:
        try:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swegym.harness.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swebench.harness.test_spec.test_spec import make_test_spec

    from openhands.events.action import CmdRunAction
    from openhands.events.observation import CmdOutputObservation

    instance["instance_id"] = instance["instance_id"].lower()
    if "version" not in instance and "base_commit" in instance:
        instance["version"] = instance["base_commit"]

    model_patch = _process_git_patch(git_patch)
    instance_id: str = instance["instance_id"]

    test_spec = make_test_spec(instance)
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
    action.set_hard_timeout(5)
    runtime.run_action(action)

    apply_cmd = (
        "cd /testbed && "
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(30)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    patch_result = obs.content  # type: ignore[attr-defined]

    if "APPLY_PATCH_FAIL" in patch_result:
        raise RuntimeError(f"{instance_id}: {APPLY_PATCH_FAIL}\n{patch_result}")
    if "APPLY_PATCH_PASS" not in patch_result:
        raise RuntimeError(
            f"{instance_id}: Unexpected output when applying patch:\n{patch_result}"
        )

    logger.debug(f"[{instance_id}] {APPLY_PATCH_PASS}:\n{patch_result}")

    log_file = "/tmp/eval_output.log"
    action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")
    action.set_hard_timeout(30)
    obs = runtime.run_action(action)
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError("Failed to launch evaluation script")

    pid = obs.content.split()[-1].strip()
    logger.debug(f"[{instance_id}] Evaluation started (PID={pid})")

    start_time = time.time()
    timeout = 20 * 60  # 20 minutes
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
        check_action = CmdRunAction(command=f"ps -p {pid} > /dev/null; echo $?")
        check_action.set_hard_timeout(5)
        check_obs = runtime.run_action(check_action)
        if (
            isinstance(check_obs, CmdOutputObservation)
            and check_obs.content.split()[-1].strip() == "1"
        ):
            logger.debug(f"[{instance_id}] Evaluation finished after {elapsed:.0f}s")
            break
        logger.debug(f"[{instance_id}] [{elapsed:.0f}s] Evaluation in progress …")
        time.sleep(30)

    cat_action = CmdRunAction(command=f"cat {log_file}")
    cat_action.set_hard_timeout(5)
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

        try:
            grading_report = get_eval_report(
                test_spec=test_spec,
                prediction={"model_patch": model_patch, "instance_id": instance_id},
                log_path=test_out_path,
                include_tests_status=True,
            )
        except Exception as e:
            if "got an unexpected keyword argument" in str(e):
                grading_report = get_eval_report(
                    test_spec=test_spec,
                    prediction={"model_patch": model_patch, "instance_id": instance_id},
                    test_log_path=test_out_path,
                    include_tests_status=True,
                )
            else:
                raise  # re-raise if it's not the error we expect
    report = grading_report[instance_id]
    logger.debug(f"[{instance_id}] Grading report: {report}")

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

async def evaluate_agent(git_patch: str | None, instance: dict, sid: str | None = None, allow_skip=True):
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
    tb = traceback.format_exc()
    instance_id = job_details.instance.get('instance_id', None) if job_details.instance is not None else None
    trajectory_id = job_details.instance.get('trajectory_id', None) if job_details.instance is not None else None
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in init: {str(e)}',
        'traceback': tb,
        'finish': False,
        'messages': [],
        'resolved': False,
        'critical_error': 'init',
    }

def run_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = job_details.instance.get('instance_id', None) if job_details.instance is not None else None
    trajectory_id = job_details.instance.get('trajectory_id', None) if job_details.instance is not None else None
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in run agent: {str(e)}',
        'traceback': tb,
        'finish': False,
        'messages': [],
        'resolved': False,
        'critical_error': 'run',
    }

def eval_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = job_details.instance.get('instance_id', None) if job_details.instance is not None else None
    trajectory_id = job_details.instance.get('trajectory_id', None) if job_details.instance is not None else None
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
        'traceback': tb,
        'finish': finish,
        'messages': messages,
        'resolved': False,
        'critical_error': 'eval',
    }

def final_result(job_details: JobDetails):
    if job_details.results is None:
        if job_details.run_results is None:
            instance_id = job_details.instance.get('instance_id', None) if job_details.instance is not None else None
            trajectory_id = job_details.instance.get('trajectory_id', None) if job_details.instance is not None else None
            result = {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'resolved': job_details.eval_results['resolved'] if job_details.eval_results is not None else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
            }
        else:
            result = {
                **job_details.run_results,
                'resolved': job_details.eval_results['resolved'] if job_details.eval_results is not None else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
            }
    else:
        result = copy.deepcopy(job_details.results)
        if job_details.timeout_error:
            result['critical_error'] = 'timeout'

    return result
