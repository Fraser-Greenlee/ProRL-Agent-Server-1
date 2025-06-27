import asyncio
import traceback
from typing import Any

import pandas as pd

from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.registry import AgentHandler, JobDetails
from openhands.runtime.base import Runtime

# Import the existing functions from utils
from openhands.nvidia.swe_agent.utils import (  # type: ignore
    initialize_agents,
    run_agent,
    evaluate_agent,
    initialize_exception,
    run_exception,
    eval_exception,
    final_result as utils_final_result,
)


class SweAgentHandler(AgentHandler):
    """Handler for SWE Agent integration, reusing functions from utils.py."""

    @property
    def name(self) -> str:
        """The name identifier for this agent handler."""
        return "swebench"

    async def init(
        self,
        instance: pd.Series,
        llm_config: LLMConfig | None = None,
        sid: str | None = None,
        max_iterations: int = 1
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the SWE Agent with instance and config using utils functions."""
        return await initialize_agents(
            instance=instance,
            llm_config=llm_config,
            sid=sid,
            max_iterations=max_iterations
        )

    async def run(
        self,
        runtime: Runtime,
        metadata: EvalMetadata,
        config: OpenHandsConfig,
        instance: pd.Series
    ) -> dict[str, object]:
        """Run the SWE Agent with runtime and instance using utils functions."""
        return await run_agent(
            runtime=runtime,
            metadata=metadata,
            config=config,
            instance=instance
        )

    async def eval(self, job_details: JobDetails, sid: str | None = None, allow_skip: bool = True) -> dict[str, Any]:
        """Evaluate the SWE Agent results using utils functions."""
        # Extract git_patch from run results
        git_patch = job_details.run_results['git_patch'] if job_details.run_results is not None else ''

        # Use the existing evaluate_agent function
        return await evaluate_agent(
            git_patch=git_patch,
            instance=job_details.instance,  # type: ignore
            sid=sid,
            allow_skip=allow_skip,
        )

    def init_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during initialization using utils functions."""
        return initialize_exception(job_details, exception)

    def run_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during run using utils functions."""
        return run_exception(job_details, exception)

    def eval_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during evaluation using utils functions."""
        return eval_exception(job_details, exception)

    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results using utils functions."""
        return utils_final_result(job_details)
