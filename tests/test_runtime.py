from __future__ import annotations

import asyncio
from pathlib import Path

from polar.runtime.docker import DockerRuntime
from polar.runtime.models import RuntimeSpec


def test_docker_upload_file_chmods_bind_mount_copy(tmp_path: Path) -> None:
    async def _run() -> None:
        runtime = DockerRuntime(
            RuntimeSpec(backend="docker", image="image"), "s1", tmp_path / "session"
        )
        runtime._chmod_needed = True
        calls: list[tuple[str, ...]] = []

        def copy_to_bind_mount(local_path: str, remote_path: str) -> bool:
            assert local_path == "local.py"
            assert remote_path == "/polar/session/workspace/calculator.py"
            return True

        async def run_local_command(
            *args: str,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            capture: bool = False,
        ) -> tuple[int, str | None, str | None]:
            calls.append(args)
            return 0, "", ""

        runtime._copy_to_bind_mount = copy_to_bind_mount  # type: ignore[method-assign]
        runtime._run_local_command = run_local_command  # type: ignore[method-assign]

        await runtime.upload_file("local.py", "/polar/session/workspace/calculator.py")

        assert calls == [
            (
                "docker",
                "exec",
                "--user",
                "root",
                "polar-s1",
                "chmod",
                "a+rwX",
                "/polar/session/workspace/calculator.py",
            )
        ]

    asyncio.run(_run())


def test_docker_upload_dir_chmods_bind_mount_copy_recursively(tmp_path: Path) -> None:
    async def _run() -> None:
        runtime = DockerRuntime(
            RuntimeSpec(backend="docker", image="image"), "s1", tmp_path / "session"
        )
        runtime._chmod_needed = True
        calls: list[tuple[str, ...]] = []

        runtime._copy_to_bind_mount = lambda local_path, remote_path: True  # type: ignore[method-assign]

        async def run_local_command(
            *args: str,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            capture: bool = False,
        ) -> tuple[int, str | None, str | None]:
            calls.append(args)
            return 0, "", ""

        runtime._run_local_command = run_local_command  # type: ignore[method-assign]

        await runtime.upload_dir("local_dir", "/polar/session/workspace/pkg")

        assert calls == [
            (
                "docker",
                "exec",
                "--user",
                "root",
                "polar-s1",
                "chmod",
                "-R",
                "a+rwX",
                "/polar/session/workspace/pkg",
            )
        ]

    asyncio.run(_run())
