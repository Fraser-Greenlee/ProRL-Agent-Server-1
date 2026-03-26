"""Singularity/Apptainer-backed rollout runtime."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from runtime.base import BaseRuntime
from runtime.models import ExecResult, RuntimeSpec


class SingularityRuntime(BaseRuntime):
    """Singularity or Apptainer instance used across rollout stages."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        safe_name = session_id.replace("/", "-")[:40]
        self._instance_name = f"arp-{safe_name}"
        self._binary = self._resolve_binary()

    @property
    def runtime_id(self) -> str:
        return self._instance_name

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("singularity runtime was already destroyed")
        args = [self._binary, "instance", "start"]
        if self.spec.network and self.spec.network != "host":
            args.extend(["--net", "--network", self.spec.network])
        # Bind-mount session dir
        args.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        args.extend([self.spec.image, self._instance_name])
        rc, _, _ = await self._run_local_command(*args)
        if rc != 0:
            raise RuntimeError(
                f"{self._binary} instance start failed with exit code {rc}"
            )

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        await self._run_local_command(
            self._binary, "instance", "stop", self._instance_name
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
        args = [self._binary, "exec", f"instance://{self._instance_name}"]
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
        # Use tar-stream through singularity exec for non-bind-mounted paths
        parent = str(Path(remote_path).parent)
        filename = Path(local_path).name
        source_dir = str(Path(local_path).parent)
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(f"failed to create directory {parent} in runtime")
        rc, _, _ = await self._run_local_command(
            "bash", "-c",
            f"tar -cf - -C {shlex.quote(source_dir)} {shlex.quote(filename)} | "
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -xf - -C {shlex.quote(parent)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"singularity upload_file failed with exit code {rc}")

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to create directory {remote_path} in runtime"
            )
        rc, _, _ = await self._run_local_command(
            "bash", "-c",
            f"tar -cf - -C {shlex.quote(local_path)} . | "
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -xf - -C {shlex.quote(remote_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"singularity upload_dir failed with exit code {rc}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        local_dir = str(Path(local_path).parent)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash", "-c",
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -cf - -C {shlex.quote(parent)} {shlex.quote(filename)} | "
            f"tar -xf - -C {shlex.quote(local_dir)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"singularity download_file failed with exit code {rc}"
            )

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash", "-c",
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -cf - -C {shlex.quote(remote_path)} . | "
            f"tar -xf - -C {shlex.quote(local_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"singularity download_dir failed with exit code {rc}"
            )

    @staticmethod
    def _resolve_binary() -> str:
        for candidate in ("singularity", "apptainer"):
            if shutil.which(candidate):
                return candidate
        return "singularity"
