from openhands.nvidia.gui_agent.gui_handler import GuiAgentHandler
from openhands.nvidia.math_coder.math_code_handler import CodeHandler, MathHandler
from openhands.nvidia.registry import add_name_mapping, register_agent_handler
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler
# from openhands.nvidia.arp.arp_handler import ARPHandler
register_agent_handler(SweAgentHandler())
register_agent_handler(MathHandler())
register_agent_handler(CodeHandler())
register_agent_handler(GuiAgentHandler())
# register_agent_handler(ARPHandler())

for code_dataset in ['codecontests', 'apps', 'codeforces', 'taco']:
    add_name_mapping(code_dataset, 'deepcoder')

# Optional aliases for GUI tasks
for gui_name in ['gui', 'visual_browsing', 'browsergym']:
    add_name_mapping(gui_name, 'gui')
