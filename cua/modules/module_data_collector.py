from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime


class ModuleDataCollector:
    """
    Class managing actual workflow to collect trajectory data.
    Orchestrates usage of EnvController and OpenAIWrapper.
    """
    def __init__(self):
        # todo
        # initialize self._init_queue and self._collect_queue
        # initialize self.job_details - storage of job

        pass

    def start_workers(self):
        """
        Start init and collect worker threads.
        """
        # todo
        # initialize self._executor (ThreadPoolExecutor)
        # initialize self._init_workers, self._collect_workers
        # submit self._run_init_worker_in_thread to self._executor
        # submit self._run_collect_worker_in_thread to self._executor
        pass

    def stop_workers(self):
        """
        Stop all workers and clean up.
        """
        # todo
        pass

    async def _init_worker(self, worker_id: int):
        """
        Init worker routine: initializes OSWorldRuntime for each worker
        """
        # todo
        # get the job_id from init_queue, and the corresponding job_detail from self.job_details
        # then boot the VM
        # then put the job_id to collect_queue so that collect_worker can fetch it
        # NOTE: use EnvController.initialize_runtime to boot up VM for this thread

        pass

    def _run_init_worker_in_thread(self, worker_id: int):
        """
        Run _init_worker in a thread with an event loop (bridging asyncio and thread pool)
        """
        # todo
        pass

    async def _collect_worker(self, worker_id: int):
        """
        Collect worker routine: collect trajectories using initialized runtimes
        """
        # todo
        # get the job_id from collect_queue,
        # then call function self.collect_trajectory to get the trajectory data
        # save the final trajectory data
        # call cleanup_job_runtime to clean things up
        # set the job_details.event so that generate_trajectories fetch it and mark completion
        pass

    async def collect_trajectory(self):
        """
        Collect a single trajectory using the provided runtime.

        Args:
            trajectory_id: Unique identifier for this trajectory
            runtime: Runtime instance to use for this trajectory
            persona: Optional persona context for this trajectory

        Returns:
            Trajectory data
        """
        # summary of previous implementation
        # loop over max_steps_per_trajectory, and in each loop,
        #   generate 1 goal
        #   for that 1 goal, we call generate_action => which updates trajectory['steps'] and returns the state after all tool-calls
        # we retain the list of goals, and the trajectory.
        # after each goal, we save the trajectory (keeps over-writing over previous save)
        # finally returns trajectory.

        # todo
        # at each goal_step in self.max_num_goals,
        # we generate 1 goal (i.e. sub-objective for the persona)
        #   input: previous goals, current screenshot, summarized result of the actions taken for previous goals (whether the goal was achieved, etc)
        #   output: a new goal
        # we generate multiple actions to achieve the goal
        #   input: the current goal, current screenshot, previous action steps taken
        #   output: a new single action
        # after each goal, save the trajectories so far (by over-writing previous save)

    def generate_goal(self):
        # summary of previous implementation
        # generate goal using LLM - maybe this could be moved to OpenAIWrapper side
        pass

    def generate_action(self):
        """
        update "steps" argument and return state, final_messages.
        """

        # summary of previous implementation
        # iterate for max_tool_calls times (inner-loop)
        #   and in each inner-loop, we generate multiple actions, and execute them
        #   across each inner-loop, we use the observations + tool call results from previous (multiple) actions as part of next input,
        #       along with the context.
        #   we do this until there's no more tool calls.
        #

        # originally, this function also called execute_action
        pass

    def _cleanup_job_runtime(self, runtime: OSWorldSingularityRuntime, job_id: str):
        """
        Cleanup runtime in background thread
        """
        # todo
        pass

    def submit_trajectory_job(self, trajectory_idx: int) -> str:
        """
        Submit a trajectory collection to the dual-queue system

        Args:
            trajectory_idx: Index of this trajectory

        Returns:
            Job ID for tracking
        """
        # todo
        # prepare job_details
        # put the job_id into init_queue so that init_worker can fetch it
        pass

    async def generate_trajectories(self):
        """
        Main routine to collect trajectories using dual-queue system
        """
        # todo
        # start workers
        # open job_ids and submit_trajectory_job
        # wait for all jobs to trigger event (in collect_worker)
        # when the event triggers, track completed_count
