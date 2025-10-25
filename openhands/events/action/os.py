from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class OSInteractiveAction(Action):
    os_actions: str
    thought: str = ''
    action: str = ActionType.OS_INTERACTIVE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'I am interacting with the operating system:\n```\n{self.os_actions}\n```'

    def __str__(self) -> str:
        ret = '**OSInteractiveAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'OS_ACTIONS: {self.os_actions}'
        return ret

