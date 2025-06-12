import time
import pandas as pd
import numpy as np
import os
import asyncio
import concurrent.futures
from evaluation.benchmarks.swe_bench.run_infer import (
    initialize_runtime,
)

from evaluation.utils.shared import (
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
from openhands.core.main import create_runtime
from openhands.core.logger import openhands_logger as logger


DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'


from evaluation.benchmarks.swe_bench.eval_infer import (
    process_git_patch as _process_git_patch,
)
from evaluation.utils.shared import EvalMetadata

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

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    SINGULARITY_IMG_DIR = (
        '/lustre/fs1/portfolios/llmservice/users/shaokunz/OpenHands_internal/'
        'singularity_images'
    )

    def _make_singularity_image_path(inst_id: str) -> str:
        image_base = 'sweb.eval.x86_64.' + inst_id.replace('__', '_s_')
        filename = f'xingyaoww_{image_base.lower()}.sif'
        return os.path.join(SINGULARITY_IMG_DIR, filename)

    sandbox_config.runtime_container_image = _make_singularity_image_path(
        instance['instance_id']
    )

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

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=False, # not too sure what this does.
    )
    config.set_agent_config(agent_config)
    return config

async def initialize_agents(
        instance: pd.Series,
        llm_config: LLMConfig | None = None,
        eval_output_dir: str = "/root",
        git_commit: str = "9f93e8a1532d6e1da4ea702f3dbd31d0f6b2fb3a",
        dataset: str = "swebench",
        data_split: str = "train",
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:

    if llm_config is None:
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            modify_params=False,
            log_completions=True,
        )

    metadata = EvalMetadata(
        agent_class="CodeActAgent",
        llm_config=llm_config,
        agent_config=None,
        max_iterations=50,
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details={'mode': 'swe'},
        condenser_config=NoOpCondenserConfig(),
    )


    config = get_config(instance, metadata)

    metadata.details['runtime_failure_count'] = 0
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config)

    await runtime.connect()

    try:
        initialize_runtime(runtime, instance, metadata)
    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    return runtime, metadata, config

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

async def _evaluate_agent(git_patch: str, instance: pd.Series):

    from openhands.utils.async_utils import call_sync_from_async

    runtime = None
    try:
        runtime, _, _ = await initialize_agents(instance)
        test_result = await call_sync_from_async(
            _apply_patch_and_evaluate, runtime, git_patch, instance
        )
    finally:
        if runtime is not None:
            try:
                runtime.event_stream.close()
            except Exception:
                pass
            await call_sync_from_async(runtime.close)

    return test_result

def _evaluate_agent_sync(git_patch: str, instance: pd.Series):

    try:
        return asyncio.run(_evaluate_agent(git_patch, instance))
    except Exception as exc:
        logger.error(f"Thread handling {instance['instance_id']} failed: {exc}")
        raise

def run_parallel_threaded(instances: list[pd.Series], git_patch: str, max_workers: int | None = None):

    if max_workers is None:
        max_workers = len(instances) if instances else 1

    results: list = [None] * len(instances)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx: dict[concurrent.futures.Future, int] = {}
        for idx, inst in enumerate(instances):
            future = pool.submit(_evaluate_agent_sync, git_patch, inst)
            future_to_idx[future] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = exc  # Preserve exception object for caller.

    return results

if __name__ == "__main__":

    dataset = pd.read_parquet("/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet")

    instance = dataset.iloc[100]['instance']
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    instance2 = dataset.iloc[0]['instance'] #0 (getmoto__moto-7365) 100 (conan-io__conan-14164) 200 (pandas-dev__pandas-54189) 150 (dask__dask-9212)
    instance2 = pd.Series(instance2)
    instance2 = instance2.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


    mock_patch = """
    diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
    new file mode 100644
    index 0000000..e69de29
    --- /dev/null
    +++ b/openhands_patch_test.txt
    @@
    +This is an OpenHands test patch.
    """

    cloned_instances: list[pd.Series] = []
    cloned_instances.append(instance)
    cloned_instances.append(instance2)

    try:
        all_reports = run_parallel_threaded(cloned_instances, mock_patch)
        for i, rep in enumerate(all_reports):
            print(f"\nBEGIN EVAL REPORT [THREAD {i}]")
            print(rep)
            print(f"END EVAL REPORT [THREAD {i}]")
    except Exception as eval_err:
        logger.error(f"Failed to threaded-evaluate patches: {eval_err}")
        raise





