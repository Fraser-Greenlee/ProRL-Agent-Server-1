import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# Use a simpler import strategy that's more compatible
try:
    # Try the most common imports first
    from jupyter_client.manager import AsyncKernelManager
    from jupyter_client.asynchronous.client import AsyncKernelClient
except ImportError:
    try:
        # Alternative import for different versions
        from jupyter_client import AsyncKernelManager
        from jupyter_client.asynchronous import AsyncKernelClient
    except ImportError:
        # Final fallback - use synchronous versions with asyncio wrappers
        from jupyter_client import KernelManager
        AsyncKernelManager = KernelManager  # type: ignore
        from jupyter_client import KernelClient
        AsyncKernelClient = KernelClient  # type: ignore

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, IPythonRunCellAction
from openhands.events.observation import IPythonRunCellObservation
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.utils.shutdown_listener import should_continue


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


@dataclass
class DirectJupyterRequirement(PluginRequirement):
    name: str = 'direct_jupyter'


class DirectJupyterKernel:
    """Direct Jupyter kernel connection without network overhead."""

    def __init__(self, kernel_id: str, lang: str = 'python') -> None:
        self.lang = lang
        self.kernel_id = kernel_id
        self.kernel_manager: Any = None  # Use Any for better compatibility
        self.kernel_client: Any = None   # Use Any for better compatibility
        self.initialized = False
        logger.info(f'Direct Jupyter kernel created for {kernel_id}')

    async def initialize(self) -> None:
        """Initialize the kernel manager and client."""
        try:
            # Create and start kernel manager
            self.kernel_manager = AsyncKernelManager(kernel_name=self.lang)

            # Handle both async and sync kernel managers
            if hasattr(self.kernel_manager, 'start_kernel'):
                if asyncio.iscoroutinefunction(self.kernel_manager.start_kernel):
                    await self.kernel_manager.start_kernel()
                else:
                    self.kernel_manager.start_kernel()

            # Get client and start channels
            self.kernel_client = self.kernel_manager.client()
            if hasattr(self.kernel_client, 'start_channels'):
                self.kernel_client.start_channels()

            # Wait for kernel to be ready
            if hasattr(self.kernel_client, 'wait_for_ready'):
                if asyncio.iscoroutinefunction(self.kernel_client.wait_for_ready):
                    await self.kernel_client.wait_for_ready(timeout=30)
                else:
                    self.kernel_client.wait_for_ready(timeout=30)

            # Initialize with no colors (same as original plugin)
            await self.execute(r'%colors nocolor')

            # pre-defined tools (same as original plugin)
            self.tools_to_run: list[str] = [
                # TODO: You can add code for your pre-defined tools here
            ]
            for tool in self.tools_to_run:
                res = await self.execute(tool)
                logger.info(f'Tool [{tool}] initialized:\n{res}')

            self.initialized = True
            logger.info('Direct Jupyter kernel initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize Direct Jupyter kernel: {e}')
            raise

    async def execute(
        self, code: str, timeout: int = 120
    ) -> Dict[str, Any]:
        """Execute code in the kernel and return structured output."""
        if not self.kernel_client or not self.kernel_manager:
            raise RuntimeError('Kernel not initialized')

        try:
            # Execute the code
            msg_id = self.kernel_client.execute(
                code,
                silent=False,
                store_history=False,
                user_expressions={},
                allow_stdin=False
            )

            logger.info(f'Executed code in direct jupyter kernel: {code[:100]}...')

            outputs: List[Dict[str, str]] = []
            execution_done = False
            start_time = time.time()

            # Collect messages until execution is complete
            while not execution_done and (time.time() - start_time) < timeout:
                try:
                    # Get messages with a short timeout to allow for overall timeout checking
                    if hasattr(self.kernel_client, 'get_iopub_msg'):
                        if asyncio.iscoroutinefunction(self.kernel_client.get_iopub_msg):
                            msg = await asyncio.wait_for(
                                self.kernel_client.get_iopub_msg(),
                                timeout=1.0
                            )
                        else:
                            # For sync version, run in executor
                            msg = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.kernel_client.get_iopub_msg(timeout=1.0)
                            )
                    else:
                        await asyncio.sleep(0.1)
                        continue

                    # Only process messages from our execution
                    if msg['parent_header'].get('msg_id') != msg_id:
                        continue

                    msg_type = msg['msg_type']
                    content = msg.get('content', {})

                    if os.environ.get('DEBUG'):
                        logger.info(
                            f'MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {content}'
                        )

                    if msg_type == 'error':
                        # Handle execution errors
                        traceback_lines = content.get('traceback', [])
                        # Remove ANSI codes from traceback
                        clean_traceback = [strip_ansi(line) for line in traceback_lines]
                        traceback_text = '\n'.join(clean_traceback)
                        outputs.append({'type': 'text', 'content': traceback_text})
                        execution_done = True

                    elif msg_type == 'stream':
                        # Handle stdout/stderr streams
                        stream_text = content.get('text', '')
                        outputs.append({'type': 'text', 'content': stream_text})

                    elif msg_type in ['execute_result', 'display_data']:
                        # Handle execution results and display data
                        data = content.get('data', {})

                        # Handle text output
                        if 'text/plain' in data:
                            text_content = data['text/plain']
                            outputs.append({'type': 'text', 'content': text_content})

                        # Handle image output (PNG)
                        if 'image/png' in data:
                            image_data = data['image/png']
                            image_url = f'data:image/png;base64,{image_data}'
                            outputs.append({'type': 'image', 'content': image_url})

                    elif msg_type == 'status':
                        # Check if execution is idle (complete)
                        if content.get('execution_state') == 'idle':
                            execution_done = True

                except asyncio.TimeoutError:
                    # Continue the loop to check overall timeout
                    continue
                except Exception as e:
                    logger.error(f'Error processing kernel message: {e}')
                    continue

            # Handle timeout
            if not execution_done:
                try:
                    # Try to interrupt the kernel
                    if self.kernel_manager and hasattr(self.kernel_manager, 'interrupt_kernel'):
                        if asyncio.iscoroutinefunction(self.kernel_manager.interrupt_kernel):
                            await self.kernel_manager.interrupt_kernel()
                        else:
                            self.kernel_manager.interrupt_kernel()
                        logger.info('Kernel interrupted due to timeout')
                except Exception as e:
                    logger.error(f'Failed to interrupt kernel: {e}')

                return {'text': f'[Execution timed out ({timeout} seconds).]', 'images': []}

            # Process collected outputs
            text_outputs = []
            image_outputs = []

            for output in outputs:
                if output['type'] == 'text':
                    text_outputs.append(output['content'])
                elif output['type'] == 'image':
                    image_outputs.append(output['content'])

            # Format final text content
            if not text_outputs and execution_done:
                text_content = '[Code executed successfully with no output]'
            else:
                text_content = ''.join(text_outputs)

            # Remove ANSI escape sequences
            text_content = strip_ansi(text_content)

            return {'text': text_content, 'images': image_outputs}

        except Exception as e:
            logger.error(f'Error executing code in direct kernel: {e}')
            return {'text': f'[Error executing code: {str(e)}]', 'images': []}

    async def shutdown_async(self) -> None:
        """Clean shutdown of kernel resources."""
        try:
            if self.kernel_client and hasattr(self.kernel_client, 'stop_channels'):
                self.kernel_client.stop_channels()
                self.kernel_client = None

            if self.kernel_manager and hasattr(self.kernel_manager, 'shutdown_kernel'):
                if asyncio.iscoroutinefunction(self.kernel_manager.shutdown_kernel):
                    await self.kernel_manager.shutdown_kernel(now=True)
                else:
                    self.kernel_manager.shutdown_kernel(now=True)
                self.kernel_manager = None

            logger.info('Direct Jupyter kernel shut down successfully')
        except Exception as e:
            logger.error(f'Error shutting down Direct Jupyter kernel: {e}')


class DirectJupyterPlugin(Plugin):
    """Direct Jupyter plugin that eliminates network overhead while maintaining compatibility."""

    name: str = 'direct_jupyter'
    kernel_id: str
    python_interpreter_path: str

    async def initialize(
        self, username: str, kernel_id: str = 'openhands-default'
    ) -> None:
        """Initialize the direct jupyter plugin."""
        self.kernel_id = kernel_id
        is_local_runtime = os.environ.get('LOCAL_RUNTIME_MODE') == '1'

        # Set up environment similar to original plugin
        if not is_local_runtime:
            # Non-LocalRuntime - set up Python path and environment
            os.environ.setdefault('POETRY_VIRTUALENVS_PATH', '/openhands/poetry')
            python_path = os.environ.get('PYTHONPATH', '')
            if '/openhands/code' not in python_path:
                os.environ['PYTHONPATH'] = f'/openhands/code:{python_path}'
            os.environ.setdefault('MAMBA_ROOT_PREFIX', '/openhands/micromamba')
        else:
            # LocalRuntime
            code_repo_path = os.environ.get('OPENHANDS_REPO_PATH')
            if not code_repo_path:
                raise ValueError(
                    'OPENHANDS_REPO_PATH environment variable is not set. '
                    'This is required for the direct jupyter plugin to work with LocalRuntime.'
                )
            # Change to the code repo directory
            os.chdir(code_repo_path)

        logger.debug('Direct Jupyter plugin initialization started')

        # Initialize the kernel directly (no network setup needed)
        self.kernel = DirectJupyterKernel(self.kernel_id)
        await self.kernel.initialize()

        # Get Python interpreter path (same as original plugin)
        _obs = await self.run(
            IPythonRunCellAction(code='import sys; print(sys.executable)')
        )
        self.python_interpreter_path = _obs.content.strip()

        logger.debug(f'Direct Jupyter plugin initialized with Python: {self.python_interpreter_path}')

    async def _run(self, action: Action) -> IPythonRunCellObservation:
        """Internal method to run a code cell in the direct jupyter kernel."""
        if not isinstance(action, IPythonRunCellAction):
            raise ValueError(
                f'Direct Jupyter plugin only supports IPythonRunCellAction, but got {action}'
            )

        if not hasattr(self, 'kernel') or not self.kernel.initialized:
            raise RuntimeError('Direct Jupyter kernel not initialized')

        # Execute the code and get structured output
        timeout = action.timeout if action.timeout is not None else 120
        if isinstance(timeout, float):
            timeout = int(timeout)

        output = await self.kernel.execute(action.code, timeout=timeout)

        # Extract text content and image URLs from the structured output
        text_content = output.get('text', '')
        image_urls = output.get('images', [])

        return IPythonRunCellObservation(
            content=text_content,
            code=action.code,
            image_urls=image_urls if image_urls else None,
        )

    async def run(self, action: Action) -> IPythonRunCellObservation:
        """Main interface for running actions."""
        obs = await self._run(action)
        return obs

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        if hasattr(self, 'kernel'):
            await self.kernel.shutdown_async()
