# type: ignore
import time
import pandas as pd
import numpy as np
import os
import asyncio
import json
import copy
import traceback
import tempfile
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
from openhands.core.setup import create_agent, create_controller
from openhands.core.main import create_runtime
from openhands.nvidia.controller import run_controller_with_controller
from openhands.controller.state.state import State
from openhands.nvidia.logger import nvidia_logger as logger

from openhands.nvidia.registry import JobDetails
from openhands.nvidia.utils import process_messages_from_agent_state, is_last_action_finish, get_messages_from_partial_result

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
    logger.debug("is_multimodal", is_multimodal)
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

def get_instance_docker_image_r2egym(instance) -> str:
    return instance["docker_image"]

def get_config(
    instance: dict,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    if 'image_assets' in instance.keys():
        is_multimodal = True
    else:
        is_multimodal = False

    if 'docker_image' in instance.keys():
        base_container_image = get_instance_docker_image_r2egym(instance)
    else:
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
    if 'docker_image' not in instance.keys():
        sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
            dataset_name=metadata.dataset or "swebench",
            instance_id=instance['instance_id'],
        )
    else:
        sandbox_config.remote_runtime_resource_factor = 1

    # run as fakeroot
    # Currently set to False as some container require GLIBC_2.38
    sandbox_config.run_as_fakeroot = False

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
    if 'docker_image' in instance.keys():
        dataset = "r2egym"

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
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    message_action = get_instruction(instance, metadata)
    try:
        agent = create_agent(config)
        job_details.agent = agent
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller
        state: State | None = await run_controller_with_controller(
                config=config,
                initial_user_action=message_action,
                sid=sid,
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
                controller=controller,
                initial_state=initial_state,
            )
        job_details.state = state

        # Try to get git patch first
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.debug(
            f'Got git diff for instance {instance['instance_id']}:\n--------\n{git_patch}\n--------'
        )

        # if fatal error, throw EvalError to log.
        if state is None:
            raise EvalException('Final state is None')
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

    except Exception as e:
        logger.error(f"Error running agent: {e}")

    # get messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state) # type: ignore
    except Exception as e:
        logger.error(f"Error while running, failed to retrieve agent messages: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    return {
        "git_patch": git_patch,
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


def _apply_patch_and_evaluate_r2egym(runtime, git_patch: str, instance: dict):

    from openhands.events.action import CmdRunAction, FileReadAction
    from openhands.events.observation import CmdOutputObservation
    import tempfile, json, re

    def _basic_pytest_parser(log: str) -> dict[str, str]:
        status_map: dict[str, str] = {}
        for line in log.split("\n"):
            if line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
                parts = line.split()
                if len(parts) >= 2:
                    status_map[parts[1].split(" - ")[0]] = parts[0]
        return status_map

    def parse_log_fn(_repo: str):  # noqa: D401
        """Return the basic pytest parser when r2egym is unavailable."""
        return _basic_pytest_parser

    def decolor_dict_keys(d: dict[str, str]) -> dict[str, str]:
        ansi_re = re.compile(r"\u001b\[[0-9;]*m")
        return {ansi_re.sub("", k): v for k, v in d.items()}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".diff") as tmp:
        tmp.write(git_patch.encode())
        tmp_path = tmp.name
    runtime.copy_to(tmp_path, "/tmp/patch.diff")

    apply_cmd = (
        "cd /testbed && "
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(60)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    patch_result = obs.content

    # Early return if patch failed to apply
    if "APPLY_PATCH_FAIL" in patch_result or "APPLY_PATCH_PASS" not in patch_result:
        return {
            "report": {
                "empty_generation": len(git_patch.strip()) == 0,
                "resolved": False,
                "failed_apply_patch": True,
                "error_eval": False,
                "test_timeout": False,
            },
            "apply_patch_output": patch_result,
            "test_output": "",
        }
    action = CmdRunAction(command="bash /root/run_tests.sh")
    action.set_hard_timeout(20 * 60)  # 20-minute hard timeout
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    test_output = obs.content
    test_output_clean = re.sub(r"\x1b\[[0-9;]*m|\r", "", test_output)
    repo_name = instance.get("repo_name") or instance.get("repo", "").split("/")[-1]
    if repo_name.endswith("_final"):
        repo_name = repo_name[: -len("_final")]

    parsed = parse_log_fn(repo_name)(test_output_clean)
    parsed = decolor_dict_keys(parsed)

    expected_json = instance.get("expected_output_json")
    if not expected_json:
        exp_obs = runtime.run_action(FileReadAction(path="/root/expected_test_output.json"))
        expected_json = getattr(exp_obs, "content", "{}")
    try:
        expected = json.loads(expected_json)
    except Exception:
        expected = {}
    expected = decolor_dict_keys(expected)
    strip_suffix = lambda d: {k.split(" - ")[0]: v for k, v in d.items()}
    parsed = strip_suffix(parsed)
    expected = strip_suffix(expected)

    resolved = bool(parsed) and len(parsed) == len(expected) and all(
        k in expected and parsed[k] == expected[k] for k in parsed
    )
    return {
        "report": {
            "empty_generation": False,
            "resolved": resolved,
            "failed_apply_patch": False,
            "error_eval": False,
            "test_timeout": False,
        },
        "apply_patch_output": patch_result,
        "test_output": test_output,
    }

def _apply_patch_and_evaluate(
    runtime,
    git_patch: str,
    instance: dict,
):

    if 'image_assets' in instance.keys():
        is_multimodal = True
    else:
        is_multimodal = False

    if 'docker_image' in instance.keys():
        return _apply_patch_and_evaluate_r2egym(runtime, git_patch, instance)

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
