from openhands.nvidia.registry import register_agent_handler


def register_swe_agent_functions():
    """Import and register swe_agent functions when needed."""
    # Import here to trigger registration decorators
    from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler

    register_agent_handler(SweAgentHandler())
