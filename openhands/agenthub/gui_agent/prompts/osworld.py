OSWORLD_OBSERVATION_FEEDBACK_PROMPT = """Action executed. Please generate the next move according to the UI screenshot and instruction.

Instruction: {instruction}

First describe the screenshot in detail, think step by step, then generate the next move. You need to at least make a tool call.
"""

ERROR_OBSERVATION_FEEDBACK_PROMPT = """Action failed. Please continue working on the task according to the instruction.
Error message: {error_message}

Instruction: {instruction}

First describe the screenshot in detail, think step by step, then generate the next move. You need to at least make a tool call.
"""