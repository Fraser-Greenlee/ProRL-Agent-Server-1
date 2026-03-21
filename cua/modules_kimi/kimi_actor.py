"""
KimiActor: single-model actor for both goal generation and action execution using Kimi-K2.5.

Reference: https://github.com/jayl940712/OSWorld/blob/fix/mm_agents/kimi/kimi_agent.py

Design: stateless — history passed as explicit parameters for thread safety.
"""
import ast
import json
from argparse import Namespace
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI

from modules_kimi.util import bytes_to_base64, bytes_to_image
from modules_kimi.util_kimi import parse_response_to_cot_and_action
from openhands.core.logger import openhands_logger

logger = openhands_logger.getChild('kimi_actor')


# ---------------------------------------------------------------------------
# Prompts / templates — copied verbatim from reference KimiAgent
# (including "passoword" typo and Unicode think markers)
# ---------------------------------------------------------------------------

# NOTE: Uses .replace("{password}", ...) instead of .format() because
# {thought}, {action}, {code} are literal placeholders for the model, not Python format strings.
SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()

SYSTEM_PROMPT_NON_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()

INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, "
    "task instruction and previous steps (if provided).\n"
)

STEP_TEMPLATE = "# Step {step_num}:\n"

THOUGHT_HISTORY_TEMPLATE_THINKING = "\u25c1think\u25b7{thought}\u25c1/think\u25b7## Action:\n{action}\n"
THOUGHT_HISTORY_TEMPLATE_NON_THINKING = "## Thought:\n{thought}\n\n## Action:\n{action}\n"


# ---------------------------------------------------------------------------
# Goal generation prompt — adapted from Planner.prepare_generate_goal_with_long_horizon
# ---------------------------------------------------------------------------
GOAL_GENERATION_PROMPT = """You are an agent generating synthetic data for OS World environment. Given example instruction(s), your task is to generate a realistic, similar-in-style, similar-in-complexity instruction suited for a new environment.

### Information on New Environment
- **OS Setup (Configuration)**:
{osworld_config}
  *Schema Reference*: `download` (file on disk), `open` (app active), `launch` (command run), `chrome_open_tabs` (browser tabs), `execute` (shell command).

- **Visual State (Screenshot)**:
  The provided screenshot represents the *current* initial state for this new task. **Crucially: If the Screenshot contradicts the OS Setup (e.g., config says a window is open, but the screen shows it closed), trust the Screenshot.**

- **Previous Requirements**:
{prev_requirements}

### Example Goals
{example_goals}

### INSTRUCTIONS
1. **Analyze Achievability**: Your goal must be achievable given the current screenshot and OS Setup. You do not have to stick to what is in the current screenshot when creating the goal - you are encouraged to leverage information not immediately present in the screenshot, but they must be present in the current OS setup. If the OS setup is empty, assume that the OS is in a clean state, thus it wouldn't make sense to refer to custom downloaded files in that case. Make sure not to hallucinate any non-existing files that cannot be found in either the screenshot or OS setup.
2. **Learn from Failures**: Review the 'Previous Requirements' section carefully.
   - The 'Verdict' explains whether a specific requirement is met in the current environment.
   - **Do not** generate a goal that relies on a condition that has already been proven impossible.
   - Use the verdict to steer your new goal toward resources that *are* available or achievable.
3. **Style & Complexity**: Generate a goal similar in style and complexity to the 'Example Goals'. Do not just copy them; adapt the logic to the files and apps visible in this environment.
4. **Define Requirements**: List the specific, **objective** pre-conditions on the environment, required for your goal to make sense in the environment. The conditions should be in a question form.
   - Ensure requirements are binary (True/False) questions, and are standalone, i.e., can be answered by inspecting the desktop environment without looking at other requirement questions.
   - Example: Does a file named `report.pdf` exist on the Desktop? (Good)
   - Example: Does the pdf file `report.pdf` on the Desktop look professional? (Bad - Subjective)

Your final response should be formatted as follows:
New Goal: [your new goal]
Requirements: ["your requirement 1", "your requirement 2", ... max 5 requirements]
"""

SPREADSHEETBENCH_GOAL_GENERATION_PROMPT = """You are an agent generating synthetic data for OS World environment. A spreadsheet file has been opened in LibreOffice Calc. Your task is to generate a concise but challenging instruction that centers on the data in this spreadsheet.

### Information on New Environment
- **OS Setup (Configuration)**:
{osworld_config}
  *Schema Reference*: `upload_file` (file on disk), `open` (app active), `launch` (command run), `execute` (shell command).

- **Visual State (Screenshot)**:
  The provided screenshot shows the spreadsheet currently open in LibreOffice Calc. Examine the visible data — column headers, data types, sheet names, and structure — to ground your instruction in the actual content.

- **Previous Requirements**:
{prev_requirements}

### Example Goals (for style and length reference)
{example_goals}

### INSTRUCTIONS
1. **Ground in the Spreadsheet**: Your instruction MUST reference the actual data visible in the screenshot (column names, sheet names, data patterns). Do not invent columns or sheets that don't exist.
2. **Length and Style**: Aim for 2-4 sentences, matching the example goals in length. Describe the end result clearly, mentioning specific columns/sheets/data, but do not dictate every click or menu navigation.
3. **Complexity**: The task should involve 2-3 distinct sub-tasks that build on each other. Pick a core spreadsheet challenge AND a meaningful follow-up — either within the spreadsheet or in another app:
   - Core challenges: cross-sheet lookups, conditional aggregation, data restructuring, formula construction, charting
   - Follow-ups that add depth: visualize the result as a chart, export a summary to Writer or Impress, save as CSV and verify in terminal, create a PDF report
   - Avoid: pure formatting tasks, single-formula tasks, or chaining 5+ unrelated steps
4. **Learn from Failures**: Review 'Previous Requirements' — do not generate a goal relying on conditions proven impossible.
5. **Define Requirements**: List specific, objective pre-conditions as binary (True/False) questions that can be verified by inspecting the environment.

Your final response should be formatted as follows:
New Goal: [your new goal]
Requirements: ["your requirement 1", "your requirement 2", ... max 5 requirements]
"""

ZENODO_GOAL_GENERATION_PROMPT = """You are an agent generating synthetic data for OS World environment. One or more presentation files (.pptx) have been opened in LibreOffice Impress. Your task is to generate a concise but challenging instruction that centers on the presentation content.

### Information on New Environment
- **OS Setup (Configuration)**:
{osworld_config}
  *Schema Reference*: `upload_file` (file on disk), `open` (app active), `launch` (command run), `execute` (shell command).

- **Visual State (Screenshot)**:
  The provided screenshot shows a presentation currently open in LibreOffice Impress. Examine the visible slides — the main slide in the center as well as other slides shown in the slide panel on the left — paying attention to titles, content, layout, images, and structure to ground your instruction in the actual content.

- **Previous Requirements**:
{prev_requirements}

### Example Goals (for style and length reference)
{example_goals}

### INSTRUCTIONS
1. **Ground in the Presentation**: Do not invent slides or content that don't exist.
2. **Length and Style**: Aim for 2-4 sentences, matching the example goals in length. Describe the end result clearly, mentioning specific slides/content, but do not dictate every click or menu navigation.
3. **Complexity**: The task should involve 2-3 distinct sub-tasks that build on each other. Pick a core presentation challenge AND a meaningful follow-up — either within the presentation or in another app:
   - Core challenges: slide restructuring, content editing across multiple slides, applying/modifying themes, adding animations or transitions, image operations, creating new slides from existing content, changing content layout, ...
   - You are encouraged to add follow-ups that add depth: export the slide, create a summary document in Writer, insert charts or tables derived from slide content, use terminal to check exported files, ...
   - Avoid: pure formatting tasks, single-slide edits, or chaining 5+ unrelated steps
4. **Learn from Failures**: Review 'Previous Requirements' — do not generate a goal relying on conditions proven impossible.
5. **Define Requirements**: List specific, objective pre-conditions as binary (True/False) questions that can be verified by inspecting the environment.
6. **Leverage Example Goals**: Try to come up with a new goal similar in style with the given example(s), while being more LibreOffice-themed and more complex.

Your final response should be formatted as follows:
New Goal: [your new goal]
Requirements: ["your requirement 1", "your requirement 2", ... max 5 requirements]
"""

OFFICE_GOAL_GENERATION_PROMPT = """You are an agent generating synthetic data for OS World environment. One or more office documents have been opened in a LibreOffice application (Calc, Impress, or Writer). Your task is to generate a concise but challenging instruction that starts from the open document and naturally involves multiple applications.

### Information on New Environment
- **OS Setup (Configuration)**:
{osworld_config}
  *Schema Reference*: `upload_file` (file on disk), `open` (app active), `launch` (command run), `execute` (shell command).

- **Visual State (Screenshot)**:
  The provided screenshot shows the current state of the desktop. Carefully examine what application is open, what document content is visible (data, slides, text, tables, images, filenames), and any other visible context. Use this to ground your instruction in what actually exists.

- **Previous Requirements**:
{prev_requirements}

### Example Goals (for style and length reference)
{example_goals}

### INSTRUCTIONS
1. **Ground in Reality**: Your instruction MUST reference actual content visible in the screenshot. Do not invent files, sheets, slides, or data that don't exist.
2. **Length and Style**: Aim for 2-4 sentences. State the desired outcome clearly without dictating step-by-step navigation.
3. **Multi-App Workflow**: The task MUST span at least 2 different applications. Start from the open document, then naturally extend to one or more other apps. The transition between apps should feel like a coherent workflow, not a forced checklist. Available apps include LibreOffice Calc/Impress/Writer, Chrome, the file manager, and the terminal.
4. **Complexity**: The task should require meaningful work in each app involved — not just opening an app and doing one trivial action. Each app step should build on or transform the output of the previous step.
5. **Learn from Failures**: Review 'Previous Requirements' — do not generate a goal relying on conditions proven impossible.
6. **Define Requirements**: List specific, objective pre-conditions as binary (True/False) questions that can be verified by inspecting the environment.

Your final response should be formatted as follows:
New Goal: [your new goal]
Requirements: ["your requirement 1", "your requirement 2", ... max 5 requirements]
"""

GOAL_PROMPT = {
    "vanilla": GOAL_GENERATION_PROMPT,
    "spreadsheetbench": OFFICE_GOAL_GENERATION_PROMPT,
    "zenodo": OFFICE_GOAL_GENERATION_PROMPT,
}


class KimiActor:
    """
    Kimi-K2.5 actor — handles both goal generation and UI action generation.

    Stateless: all history is passed as explicit parameters.
    """

    def __init__(self, args: Namespace):
        self.client = OpenAI(
            base_url=f"http://{args.model_node}:8000/v1",
            api_key="gen",
        )
        self.model_name = args.kimi_model_name
        self.generation_mode = args.generation_mode

        # Defaults from reference
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 4096
        self.thinking = True
        self.password = "password"
        self.screen_size = (1920, 1080)
        self.max_image_history_length = 3
        self.max_retries = 5
        self.coordinate_type = "relative"

    # ------------------------------------------------------------------
    # Goal generation
    # ------------------------------------------------------------------

    def generate_goal(
        self,
        screenshot_bytes: bytes,
        osworld_config: List[Dict],
        example_goals: List[str],
        prev_requirements: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Generate a high-level goal from the current screenshot and OSWorld config.

        Returns:
            (goal, requirements) or (None, None) on failure.
        """
        if prev_requirements:
            prev_req_str = "\n".join(
                f"- Condition: {c} | Verdict: {v}" for c, v in prev_requirements
            )
        else:
            prev_req_str = "None (Initial Attempt)"

        example_goals_str = "\n".join(f"- {g}" for g in example_goals)

        prompt_text = GOAL_PROMPT[self.generation_mode].format(
            osworld_config=json.dumps(osworld_config, indent=4),
            prev_requirements=prev_req_str,
            example_goals=example_goals_str,
        )

        image_url = bytes_to_base64(screenshot_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                result = self._call_llm(messages, temperature=1.0)
                content = result.get("content", "")

                goal, requirements = self._parse_goal_response(content)
                if goal is not None and requirements is not None:
                    return goal, requirements
            except Exception as e:
                logger.warning(f"generate_goal attempt {attempt + 1} failed: {e}")

        logger.warning("generate_goal exhausted all retries.")
        return None, None

    @staticmethod
    def _parse_goal_response(generation: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Parse goal generation response. Reuses logic from Planner.parse_with_long_horizon.
        Handles chain-of-thought (</think>), reasoning markers, and various list formats.
        """
        # Strip chain-of-thought if present
        if "</think>" in generation:
            generation = generation.split("</think>")[-1]

        if "New Goal: " not in generation or "Requirements: " not in generation:
            return None, None

        try:
            content_start = generation.index("New Goal: ")
            clean_generation = generation[content_start:]

            part1 = clean_generation.split("Requirements: ", 1)
            if len(part1) != 2:
                return None, None

            raw_goal_str, raw_reqs_str = part1

            goal = raw_goal_str.replace("New Goal: ", "").strip()

            raw_reqs_str = raw_reqs_str.strip()
            if raw_reqs_str.startswith("```"):
                lines = raw_reqs_str.splitlines()
                if len(lines) >= 3:
                    raw_reqs_str = "\n".join(lines[1:-1])
                else:
                    raw_reqs_str = raw_reqs_str.strip("`").replace("json", "")

            try:
                requirements = json.loads(raw_reqs_str)
            except json.JSONDecodeError:
                try:
                    requirements = ast.literal_eval(raw_reqs_str)
                except (ValueError, SyntaxError):
                    requirements = [
                        line.strip("- *") for line in raw_reqs_str.splitlines() if line.strip()
                    ]

            if not isinstance(requirements, list):
                return None, None

            if len(goal) >= 1000:
                return None, None

            requirements = [
                str(r) for r in requirements if isinstance(r, (str, int, float))
            ]

            return goal, requirements

        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def generate_action(
        self,
        instruction: str,
        screenshot_bytes: bytes,
        history_screenshots: List[bytes],
        history_actions: List[str],
        history_cots: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate the next UI action given instruction, current screenshot, and history.

        Args:
            instruction: The goal/instruction to execute.
            screenshot_bytes: Current screenshot as raw bytes.
            history_screenshots: List of previous screenshot bytes.
            history_actions: List of previous action description strings.
            history_cots: List of previous cot dicts (with "thought" and "action" keys).

        Returns:
            Dict with keys:
                "pyautogui_command": str — code to execute on the VM
                "action_type": "pyautogui" | "wait" | "done" | "fail"
                "action_generation": {"thought": "...", "action": "...", "original_code": "...", "code": "..."}
            Or None on failure.
        """
        messages = self._build_messages(
            instruction, screenshot_bytes, history_screenshots, history_actions, history_cots
        )

        temperature = self.temperature
        for attempt in range(self.max_retries):
            try:
                result = self._call_llm(messages, temperature=temperature)

                if not result.get("content"):
                    raise ValueError("Empty response from LLM")

                low_level_instruction, pyautogui_actions, sections = parse_response_to_cot_and_action(
                    response={
                        "content": result.get("content", ""),
                        "reasoning_content": result.get("reasoning", ""),
                    },
                    screen_size=self.screen_size,
                    coordinate_type=self.coordinate_type,
                    thinking=self.thinking,
                )

                if "<Error>" in low_level_instruction or not pyautogui_actions:
                    raise ValueError(f"Error parsing response: {low_level_instruction}")

                # Determine action type and command from parsed actions list
                action_code = pyautogui_actions[0]

                if action_code == "WAIT":
                    action_type = "wait"
                    pyautogui_command = "import time\ntime.sleep(20)"
                elif action_code == "DONE":
                    action_type = "done"
                    pyautogui_command = ""
                elif action_code == "FAIL":
                    action_type = "fail"
                    pyautogui_command = ""
                else:
                    action_type = "pyautogui"
                    pyautogui_command = action_code

                return {
                    "pyautogui_command": pyautogui_command,
                    "action_type": action_type,
                    "action_generation": sections,
                    "raw_response": result.get("content", ""),
                    "raw_reasoning": result.get("reasoning", ""),
                }

            except Exception as e:
                logger.warning(f"generate_action attempt {attempt + 1} failed: {e}")

            # Temperature fallback per reference
            temperature = max(0.2, temperature)

        logger.warning("generate_action exhausted all retries.")
        return None

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict], temperature: float) -> Dict[str, str]:
        """
        Call vLLM via OpenAI client. Returns dict with "content" and "reasoning" keys.
        """
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            timeout=900,
        )

        message = chat_response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "") or ""

        return {"content": content, "reasoning": reasoning}

    # ------------------------------------------------------------------
    # Message construction — matches reference predict() exactly
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        instruction: str,
        screenshot_bytes: bytes,
        history_screenshots: List[bytes],
        history_actions: List[str],
        history_cots: List[Dict],
    ) -> List[Dict]:
        """
        Build the message list for Kimi-K2.5, following the reference's exact windowing logic.

        - Old steps (beyond max_image_history_length): concatenated into a single
          text-only assistant message.
        - Recent steps: user message with screenshot image, then assistant message
          with step template + history template.
        - Final message: current screenshot + instruction.
        """
        if self.thinking:
            system_prompt = SYSTEM_PROMPT_THINKING.replace("{password}", self.password)
            history_template = THOUGHT_HISTORY_TEMPLATE_THINKING
        else:
            system_prompt = SYSTEM_PROMPT_NON_THINKING.replace("{password}", self.password)
            history_template = THOUGHT_HISTORY_TEMPLATE_NON_THINKING

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        num_steps = len(history_actions)
        instruction_prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)

        # Build history messages following reference's exact windowing logic
        history_step_texts = []
        for i in range(num_steps):
            if i > num_steps - self.max_image_history_length:
                # Recent step: include screenshot image
                image_url = bytes_to_base64(history_screenshots[i])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                })

                history_content = STEP_TEMPLATE.format(step_num=i + 1) + history_template.format(
                    thought=history_cots[i].get('thought', ''),
                    action=history_cots[i].get('action', ''),
                )
                messages.append({
                    "role": "assistant",
                    "content": history_content,
                })
            else:
                # Old step: text-only, accumulate
                history_content = STEP_TEMPLATE.format(step_num=i + 1) + history_template.format(
                    thought=history_cots[i].get('thought', ''),
                    action=history_cots[i].get('action', ''),
                )
                history_step_texts.append(history_content)

                # Emit accumulated text as single assistant message at the boundary
                if i == num_steps - self.max_image_history_length:
                    messages.append({
                        "role": "assistant",
                        "content": "\n".join(history_step_texts),
                    })

        # Final user message: current screenshot + instruction
        current_image_url = bytes_to_base64(screenshot_bytes)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": current_image_url}},
                {"type": "text", "text": instruction_prompt},
            ],
        })

        return messages
