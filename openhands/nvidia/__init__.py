from openhands.nvidia.math_coder.math_code_handler import MathHandler
from openhands.nvidia.registry import register_agent_handler
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler

register_agent_handler(SweAgentHandler())
register_agent_handler(MathHandler())
