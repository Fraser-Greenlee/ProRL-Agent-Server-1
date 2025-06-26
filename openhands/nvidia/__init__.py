def register_swe_agent_functions():
    """Import and register swe_agent functions when needed."""
    # Import here to trigger registration decorators
    from openhands.nvidia.swe_agent.utils import (
        eval_exception,
        evaluate_agent,
        final_result,
        initialize_agents,
        initialize_exception,
        run_agent,
        run_exception,
    )
    # Return the functions if needed elsewhere
    return {
        'eval_exception': eval_exception,
        'evaluate_agent': evaluate_agent,
        'final_result': final_result,
        'initialize_agents': initialize_agents,
        'initialize_exception': initialize_exception,
        'run_agent': run_agent,
        'run_exception': run_exception,
    }
