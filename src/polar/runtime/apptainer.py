"""Apptainer-backed rollout runtime."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import shutil
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


class ApptainerRuntime(BaseRuntime):
    """Apptainer instance used across rollout stages."""

    _INSTANCE_COMMAND_LOCK = asyncio.Lock()
    _START_ATTEMPTS = 4

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        # Use a hash suffix to guarantee uniqueness even when session IDs
        # share a long prefix (e.g. "sk-polar-...-eval" vs "sk-polar-...").
        short_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        safe_name = session_id.replace("/", "-")[:30]
        self._instance_name = f"polar-{safe_name}-{short_hash}"
        self._binary = self._resolve_binary()
        self._overlay_dir: Path | None = None
        self._use_instance = os.environ.get("POLAR_APPTAINER_NO_INSTANCE", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def runtime_id(self) -> str:
        return self._instance_name

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return True

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("apptainer runtime was already destroyed")
        # Use a host-backed overlay directory instead of --writable-tmpfs
        # (default tmpfs overlay is only 64 MB, too small for most workloads).
        self._overlay_dir = self.session_dir / "overlay"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)
        runtime_options = self._runtime_options()
        if not self._use_instance:
            logger.info(
                "Using direct apptainer exec runtime for %s",
                self._instance_name,
            )
            rc, _, stderr = await self._run_local_command(
                self._binary,
                "exec",
                *runtime_options,
                self.spec.image,
                "true",
                capture=True,
            )
            if rc != 0:
                message = f"{self._binary} direct exec validation failed with exit code {rc}"
                if stderr:
                    message = f"{message}: {stderr.strip()}"
                raise RuntimeError(message)
            return

        args = [
            self._binary,
            "instance",
            "start",
            *runtime_options,
            self.spec.image,
            self._instance_name,
        ]
        last_rc = 0
        last_stderr: str | None = None
        for attempt in range(1, self._START_ATTEMPTS + 1):
            logger.info(
                "Starting apptainer instance %s (attempt %d/%d)",
                self._instance_name,
                attempt,
                self._START_ATTEMPTS,
            )
            async with self._INSTANCE_COMMAND_LOCK:
                last_rc, _, last_stderr = await self._run_local_command(
                    *args, capture=True
                )
            if last_rc == 0:
                logger.info("Started apptainer instance %s", self._instance_name)
                return
            if attempt < self._START_ATTEMPTS:
                logger.warning(
                    "%s instance start failed for %s (attempt %d/%d, rc=%s): %s",
                    self._binary,
                    self._instance_name,
                    attempt,
                    self._START_ATTEMPTS,
                    last_rc,
                    (last_stderr or "").strip(),
                )
                await asyncio.sleep(min(2.0 * attempt, 8.0))

        message = f"{self._binary} instance start failed with exit code {last_rc}"
        if last_stderr:
            message = f"{message}: {last_stderr.strip()}"
        raise RuntimeError(message)

    _STOP_TIMEOUT = 30.0

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if not self._use_instance:
            return
        async with self._INSTANCE_COMMAND_LOCK:
            rc, _, stderr = await self._run_local_command(
                self._binary, "instance", "stop", self._instance_name,
                timeout=self._STOP_TIMEOUT, capture=True,
            )
        if rc != 0:
            logger.warning(
                "%s instance stop failed for %s (rc=%s): %s",
                self._binary, self._instance_name, rc, stderr,
            )

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        wrapped_command = command
        if effective_workdir:
            wrapped_command = f"cd {shlex.quote(effective_workdir)} && {command}"
        args = self._exec_base_args()
        if env:
            args.append("env")
            args.extend(f"{key}={value}" for key, value in env.items())
        args.extend(["bash", "-lc", wrapped_command])
        rc, stdout, stderr = await self._run_local_command(
            *args, timeout=timeout_sec, capture=True
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(local_path).name
        source_dir = str(Path(local_path).parent)
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(f"failed to create directory {parent} in runtime")
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(source_dir)} {shlex.quote(filename)} | "
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -xf - -C {shlex.quote(parent)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_file failed with exit code {rc}")

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to create directory {remote_path} in runtime"
            )
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(local_path)} . | "
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -xf - -C {shlex.quote(remote_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_dir failed with exit code {rc}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        local_dir = str(Path(local_path).parent)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -cf - -C {shlex.quote(parent)} {shlex.quote(filename)} | "
            f"tar -xf - -C {shlex.quote(local_dir)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_file failed with exit code {rc}"
            )

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -cf - -C {shlex.quote(remote_path)} . | "
            f"tar -xf - -C {shlex.quote(local_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_dir failed with exit code {rc}"
            )

    def _runtime_options(self) -> list[str]:
        if self._overlay_dir is None:
            raise RuntimeError("apptainer runtime overlay was not initialized")
        args = ["--overlay", str(self._overlay_dir)]
        if self.spec.gpus > 0:
            args.append("--nv")
        network_name = "none" if not self.spec.allow_internet else self.spec.network
        if network_name and network_name != "host":
            args.extend(["--net", "--network", network_name])
        args.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        # Additional bind mounts from kwargs (mirrors Docker's volumes support).
        for vol in self.spec.kwargs.get("volumes", []):
            # Accept Docker-style "host:container[:ro]" strings.
            args.extend(["--bind", vol])
        return args

    def _exec_base_args(self) -> list[str]:
        if self._use_instance:
            return [self._binary, "exec", f"instance://{self._instance_name}"]
        return [self._binary, "exec", *self._runtime_options(), self.spec.image]

    @staticmethod
    def _shell_join(args: list[str]) -> str:
        return " ".join(shlex.quote(arg) for arg in args)

    @staticmethod
    def _resolve_binary() -> str:
        override = os.environ.get("POLAR_APPTAINER_BIN")
        if override:
            return override
        for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
            if Path(candidate).is_file():
                return candidate
        resolved = shutil.which("apptainer")
        if resolved:
            return resolved
        return "apptainer"
