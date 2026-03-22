import atexit
import asyncio
import json
import logging
import os
import platform
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from filelock import FileLock

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.nvcf.NVCFProvider")
logger.setLevel(logging.INFO)

NVCF_BASE_URL = "https://grpc.nvcf.nvidia.com"
NGC_ORG = os.environ.get("NGC_ORG")
DEFAULT_CONTAINER_IMAGE = (
    f"nvcr.io/{NGC_ORG}/nemo:osworld-linux-eval-v2" if NGC_ORG else None
)
DEFAULT_LOCAL_PROXY_PORT = 8090
RETRY_INTERVAL = 10
LOCK_TIMEOUT = 30
HEALTH_PING_INTERVAL = 30
HEALTH_PING_TIMEOUT = 30


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _find_available_port(start_port: int) -> int:
    port = max(1, int(start_port))
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError("No available local ports found for NVCF proxy")


class NVCFLocalProxy:
    """
    Local HTTP/WebSocket proxy for NVCF.

    Requests to:
      - /api/*     -> https://grpc.nvcf.nvidia.com/api/*
      - /chrome/*  -> https://grpc.nvcf.nvidia.com/chrome/*
      - /vlc/*     -> https://grpc.nvcf.nvidia.com/vlc/*

    are forwarded with the required NVCF auth headers.
    """

    def __init__(
        self,
        api_key: str,
        function_id: str,
        local_host: str = "127.0.0.1",
        local_port: int = DEFAULT_LOCAL_PROXY_PORT,
    ):
        self.api_key = api_key
        self.function_id = function_id
        self.local_host = local_host
        self.local_port = int(local_port)

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._ready = threading.Event()
        self._start_error: Optional[Exception] = None
        self._running = False

    @property
    def local_url(self) -> str:
        return f"http://{self.local_host}:{self.local_port}"

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Function-ID": self.function_id,
            "Host": "grpc.nvcf.nvidia.com",
        }

    def _rewrite_chrome_json(self, path: str, content_type: str, body: bytes) -> bytes:
        if not path.startswith("/chrome"):
            return body

        if "application/json" not in (content_type or ""):
            return body

        try:
            text = body.decode("utf-8")
        except Exception:
            return body

        # Chrome discovery often returns ws://localhost:9222/devtools/...
        # Rewrite all localhost/loopback websocket targets to the local proxy.
        ws_base = f"ws://{self.local_host}:{self.local_port}/chrome/"
        pattern = re.compile(r"ws://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?/")
        rewritten = pattern.sub(ws_base, text)

        # Keep frontend inspector links coherent as well (contains `ws=host:port/...`).
        rewritten = re.sub(
            r"ws=(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?/",
            f"ws={self.local_host}:{self.local_port}/chrome/",
            rewritten,
        )

        # Best effort JSON roundtrip for stable formatting when possible.
        try:
            parsed = json.loads(rewritten)
            return json.dumps(parsed).encode("utf-8")
        except Exception:
            return rewritten.encode("utf-8")

    async def _pipe_ws(self, request: web.Request, target_url: str) -> web.StreamResponse:
        ws_local = web.WebSocketResponse(autoping=True)
        await ws_local.prepare(request)

        headers = self._auth_headers()
        for key, value in request.headers.items():
            if key.lower() in {"host", "connection", "upgrade", "authorization", "function-id"}:
                continue
            headers[key] = value

        timeout = ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
        ws_target = target_url.replace("https://", "wss://", 1)

        async with ClientSession(timeout=timeout) as session:
            async with session.ws_connect(ws_target, headers=headers) as ws_remote:
                async def local_to_remote():
                    async for msg in ws_local:
                        if msg.type == WSMsgType.TEXT:
                            await ws_remote.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await ws_remote.send_bytes(msg.data)
                        elif msg.type == WSMsgType.PING:
                            await ws_remote.ping()
                        elif msg.type == WSMsgType.PONG:
                            await ws_remote.pong()
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                            await ws_remote.close()
                            break

                async def remote_to_local():
                    async for msg in ws_remote:
                        if msg.type == WSMsgType.TEXT:
                            await ws_local.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await ws_local.send_bytes(msg.data)
                        elif msg.type == WSMsgType.PING:
                            await ws_local.ping()
                        elif msg.type == WSMsgType.PONG:
                            await ws_local.pong()
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                            await ws_local.close()
                            break

                await asyncio.gather(local_to_remote(), remote_to_local())

        return ws_local

    async def _proxy_handler(self, request: web.Request) -> web.StreamResponse:
        target_url = f"{NVCF_BASE_URL}{request.path_qs}"
        is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
        if is_ws:
            return await self._pipe_ws(request, target_url)

        body = await request.read()
        headers = self._auth_headers()
        for key, value in request.headers.items():
            if key.lower() in {
                "host",
                "content-length",
                "connection",
                "authorization",
                "function-id",
            }:
                continue
            headers[key] = value

        timeout = ClientTimeout(total=120, connect=30, sock_connect=30, sock_read=120)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.request(
                    request.method,
                    target_url,
                    data=body if body else None,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    response_body = await resp.read()
                    content_type = resp.headers.get("Content-Type", "")
                    response_body = self._rewrite_chrome_json(
                        request.path,
                        content_type,
                        response_body,
                    )
                    out = web.Response(status=resp.status, body=response_body)
                    excluded_headers = {
                        "content-length",
                        "transfer-encoding",
                        "connection",
                        "content-encoding",
                    }
                    for key, value in resp.headers.items():
                        if key.lower() in excluded_headers:
                            continue
                        out.headers[key] = value
                    return out
        except Exception as e:
            logger.error("NVCF proxy request failed for %s: %s", request.path_qs, e)
            return web.Response(status=502, text=f"NVCF proxy error: {e}")

    async def _async_start(self):
        app = web.Application(client_max_size=256 * 1024**2)
        app.router.add_route("*", "/{path:.*}", self._proxy_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.local_host, self.local_port)
        await self._site.start()
        self._running = True

    async def _async_stop(self):
        self._running = False
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
        except Exception as e:
            self._start_error = e
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        # Cleanup if loop stops unexpectedly
        try:
            self._loop.run_until_complete(self._async_stop())
        except Exception:
            pass
        self._loop.close()

    def start(self, timeout: int = 10):
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for local NVCF proxy to start")
        if self._start_error:
            raise RuntimeError(f"Failed to start NVCF proxy: {self._start_error}")
        logger.info("NVCF proxy started at %s", self.local_url)

    def stop(self, timeout: int = 10):
        if not self._thread:
            return
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning("Failed to cleanly stop NVCF proxy: %s", e)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None
        self._site = None
        self._runner = None
        self._running = False
        logger.info("NVCF proxy stopped")


@dataclass
class NVCFFunctionConfig:
    name: str
    container_image: str = field(
        default_factory=lambda: os.environ.get("NVCF_CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE)
    )
    inference_url: str = "/api"
    inference_port: int = 8000
    health_uri: str = "/api/version"
    description: str = "OSWorld NVCF runtime managed by OSWorld provider"
    container_args: Optional[str] = None
    container_environment_variables: Optional[list[str]] = None
    tags: Optional[list[str]] = field(default_factory=lambda: ["osworld", "nvcf"])


@dataclass
class NVCFDeploymentConfig:
    gpu: str = "L40S"
    instance_type: Optional[str] = None
    min_instances: int = 1
    max_instances: int = 1
    max_request_concurrency: int = 1
    backend: str = "GFN"
    regions: Optional[list[str]] = None
    clusters: Optional[list[str]] = None
    attributes: Optional[list[str]] = None
    configuration: Optional[dict] = None

    def get_instance_type(self) -> str:
        if self.instance_type:
            return self.instance_type
        defaults = {
            ("GFN", "L40"): "gl40_1.br20_2xlarge",
            ("GFN", "L40G"): "gl40g_1.br25_2xlarge",
            ("GFN", "L40S"): "gl40s_4.br25_small"
        }
        return defaults.get((self.backend, self.gpu), f"{self.backend}.GPU.{self.gpu}_1x")


class NVCFDeployer:
    def __init__(self, api_key: str, org_name: str, team_name: str = "no-team"):
        try:
            from ngcsdk import Client
            from nvcf.api.deployment_spec import TargetedDeploymentSpecification
        except ImportError as e:
            raise ImportError(
                "ngcsdk is required for NVCF auto-deploy. Install with: pip install ngcsdk"
            ) from e

        self._TargetedDeploymentSpecification = TargetedDeploymentSpecification
        self._api_key = api_key
        self._client = Client()
        self._client.configure(api_key, org_name=org_name, team_name=team_name)

    def create_function(self, config: NVCFFunctionConfig) -> dict:
        kwargs = {
            "name": config.name,
            "inference_url": config.inference_url,
            "container_image": config.container_image,
            "inference_port": config.inference_port,
            "health_uri": config.health_uri,
            "description": config.description,
        }
        if config.container_args:
            kwargs["container_args"] = config.container_args
        if config.container_environment_variables:
            kwargs["container_environment_variables"] = config.container_environment_variables
        if config.tags:
            kwargs["tags"] = config.tags
        return self._client.cloud_function.functions.create(**kwargs)

    def deploy(self, function_id: str, version_id: str, config: NVCFDeploymentConfig) -> dict:
        spec = self._TargetedDeploymentSpecification(
            gpu=config.gpu,
            instance_type=config.get_instance_type(),
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            max_request_concurrency=config.max_request_concurrency,
            regions=config.regions,
            clusters=config.clusters,
            attributes=config.attributes,
            configuration=config.configuration,
        )
        return self._client.cloud_function.functions.deployments.create(
            function_id=function_id,
            function_version_id=version_id,
            targeted_deployment_specifications=[spec],
        )

    def get_status(self, function_id: str, version_id: str) -> str:
        info = self._client.cloud_function.functions.info(
            function_id=function_id,
            function_version_id=version_id,
        )
        return info.get("function", {}).get("status", "UNKNOWN")

    def wait_for_active(
        self,
        function_id: str,
        version_id: str,
        timeout: int = 1800,
        poll_interval: int = 10,
    ) -> str:
        start = time.time()
        status = "UNKNOWN"
        while time.time() - start < timeout:
            status = self.get_status(function_id, version_id)
            if status == "ACTIVE":
                return status
            if status == "ERROR":
                # Clean up the failed function before raising
                try:
                    logger.info("NVCF function entered ERROR state, cleaning up %s/%s...", function_id, version_id)
                    self.delete_function(function_id, version_id)
                except Exception as e:
                    logger.warning("Failed to delete ERROR function %s/%s: %s", function_id, version_id, e)
                raise RuntimeError("NVCF function entered ERROR state")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"NVCF function did not become ACTIVE within {timeout}s. Last status: {status}"
        )

    def undeploy(self, function_id: str, version_id: str):
        self._client.cloud_function.functions.deployments.delete(
            function_id=function_id,
            function_version_id=version_id,
            graceful=True,
        )

    def delete_function(self, function_id: str, version_id: str):
        self._client.cloud_function.functions.delete(
            function_id=function_id,
            function_version_id=version_id,
        )


class NVCFProvider(Provider):
    """
    NVCF-backed provider with optional auto deploy/undeploy and local proxy.

    By default:
      - If NVCF_FUNCTION_ID exists, attach to it.
      - Otherwise create+deploy a new function (requires ngcsdk and NGC_ORG).
      - Always expose one local proxy port used for /api, /chrome and /vlc.
    """

    def __init__(self, region: str = None):
        super().__init__(region)
        self._proxy: Optional[NVCFLocalProxy] = None
        self._deployer: Optional[NVCFDeployer] = None
        self._we_deployed_this = False
        self._final_cleanup_done = False
        self._health_thread: Optional[threading.Thread] = None
        self._health_stop_event = threading.Event()

        self._api_key: Optional[str] = None
        self._org_name: Optional[str] = None
        self._function_id: Optional[str] = None
        self._version_id: Optional[str] = None

        self.server_port: Optional[int] = None
        self.chromium_port: Optional[int] = None
        self.vlc_port: Optional[int] = None
        self.vnc_port: Optional[int] = None

        temp_dir = Path(os.getenv("TEMP") if platform.system() == "Windows" else "/tmp")
        self._port_lock_file = temp_dir / "nvcf_proxy_port_allocation.lck"
        self._port_lock_file.parent.mkdir(parents=True, exist_ok=True)
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        try:
            self._stop_emulator(final=True)
        except Exception:
            pass

    def __del__(self):
        try:
            self._stop_emulator(final=True)
        except Exception:
            pass

    def _ensure_deployer(self):
        if self._deployer is not None:
            return
        self._org_name = os.environ.get("NGC_ORG")
        if not self._org_name:
            raise ValueError("NGC_ORG is required for NVCF provider")
        team_name = os.environ.get("NGC_TEAM", "no-team")
        self._deployer = NVCFDeployer(
            api_key=self._api_key,
            org_name=self._org_name,
            team_name=team_name,
        )

    def _build_deployment_config(self) -> NVCFDeploymentConfig:
        return NVCFDeploymentConfig(
            gpu=os.environ.get("NVCF_GPU", "L40S"),
            instance_type=os.environ.get("NVCF_INSTANCE_TYPE") or None,
            min_instances=int(os.environ.get("NVCF_MIN_INSTANCES", "1")),
            max_instances=int(os.environ.get("NVCF_MAX_INSTANCES", "1")),
            max_request_concurrency=int(os.environ.get("NVCF_MAX_REQUEST_CONCURRENCY", "1")),
            backend=os.environ.get("NVCF_BACKEND", "GFN"),
        )

    def _health_monitor_loop(self):
        while not self._health_stop_event.is_set():
            if not self.server_port:
                break
            url = f"http://127.0.0.1:{self.server_port}/api/version"
            try:
                response = requests.get(url, timeout=HEALTH_PING_TIMEOUT)
                if response.status_code != 200:
                    logger.error(
                        "NVCF health check failed for function %s: status=%s url=%s",
                        self._function_id,
                        response.status_code,
                        url,
                    )
            except Exception as e:
                logger.error(
                    "NVCF health check error for function %s: %s (url=%s)",
                    self._function_id,
                    e,
                    url,
                )
            if self._health_stop_event.wait(HEALTH_PING_INTERVAL):
                break

    def _start_health_monitor(self):
        if self._health_thread is not None and self._health_thread.is_alive():
            return
        self._health_stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name=f"nvcf-health-{os.getpid()}",
        )
        self._health_thread.start()

    def _stop_health_monitor(self, timeout: int = 5):
        self._health_stop_event.set()
        if self._health_thread is not None and self._health_thread.is_alive():
            self._health_thread.join(timeout=timeout)
        self._health_thread = None

    def _create_and_deploy_new_function(self, deployment_config: NVCFDeploymentConfig):
        name_prefix = os.environ.get("NVCF_FUNCTION_NAME_PREFIX", "osworld-nvcf")
        function_name = f"{name_prefix}-{os.getpid()}-{int(time.time())}"
        image = os.environ.get("NVCF_CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE)
        function_config = NVCFFunctionConfig(name=function_name, container_image=image)

        logger.info("Creating NVCF function with image: %s", image)
        result = self._deployer.create_function(function_config)
        function = result.get("function", {})
        self._function_id = function.get("id")
        self._version_id = function.get("versionId")
        if not self._function_id or not self._version_id:
            raise RuntimeError(f"NVCF create_function did not return IDs: {result}")

        logger.info(
            "Deploying NVCF function %s (version %s)...",
            self._function_id,
            self._version_id,
        )
        self._deployer.deploy(self._function_id, self._version_id, deployment_config)
        self._deployer.wait_for_active(
            self._function_id,
            self._version_id,
            timeout=int(os.environ.get("NVCF_DEPLOY_TIMEOUT", "600")),
            poll_interval=int(os.environ.get("NVCF_DEPLOY_POLL_INTERVAL", "10")),
        )
        self._we_deployed_this = True
        logger.info("NVCF function is ACTIVE: %s", self._function_id)

    def _deploy_if_needed(self):
        existing_function_id = os.environ.get("NVCF_FUNCTION_ID")
        existing_version_id = os.environ.get("NVCF_VERSION_ID")
        if existing_function_id:
            self._function_id = existing_function_id.strip()
            self._version_id = existing_version_id.strip() if existing_version_id else None
            logger.info("Using existing NVCF function: %s", self._function_id)
            self._we_deployed_this = False
            return

        self._ensure_deployer()
        deployment_config = self._build_deployment_config()

        if self._function_id and self._version_id and self._we_deployed_this:
            try:
                status = self._deployer.get_status(self._function_id, self._version_id)
                if status == "ACTIVE":
                    logger.info(
                        "Reusing ACTIVE NVCF function %s (version %s).",
                        self._function_id,
                        self._version_id,
                    )
                    return
            except Exception as e:
                logger.warning("Failed to fetch existing NVCF function status: %s", e)

            try:
                logger.info(
                    "Redeploying existing NVCF function %s (version %s)...",
                    self._function_id,
                    self._version_id,
                )
                self._deployer.deploy(self._function_id, self._version_id, deployment_config)
                self._deployer.wait_for_active(
                    self._function_id,
                    self._version_id,
                    timeout=int(os.environ.get("NVCF_DEPLOY_TIMEOUT", "600")),
                    poll_interval=int(os.environ.get("NVCF_DEPLOY_POLL_INTERVAL", "10")),
                )
                logger.info("NVCF function is ACTIVE: %s", self._function_id)
                return
            except Exception as e:
                logger.warning(
                    "Failed to redeploy existing NVCF function %s (version %s): %s. Creating a new function.",
                    self._function_id,
                    self._version_id,
                    e,
                )
                self._function_id = None
                self._version_id = None
                self._we_deployed_this = False

        self._create_and_deploy_new_function(deployment_config)

    def _wait_for_vm_ready(self, timeout: int = 300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.server_port}/api/screenshot",
                    timeout=(10, 10),
                )
                if response.status_code == 200:
                    return
            except Exception:
                pass
            logger.info("Checking if NVCF proxy endpoint is ready...")
            time.sleep(RETRY_INTERVAL)
        raise TimeoutError("NVCF endpoint failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        del path_to_vm, headless, os_type

        # Allow new lifecycle after prior final cleanup.
        self._final_cleanup_done = False

        self._api_key = os.environ.get("NGC_API_KEY")
        if not self._api_key:
            raise ValueError("NGC_API_KEY is required for NVCF provider")
        global NGC_ORG, DEFAULT_CONTAINER_IMAGE
        NGC_ORG = os.environ.get("NGC_ORG")
        if not NGC_ORG:
            raise ValueError("NGC_ORG is required for NVCF provider")
        DEFAULT_CONTAINER_IMAGE = f"nvcr.io/{NGC_ORG}/nemo:osworld-linux-eval-v2"

        self._deploy_if_needed()
        try:
            lock = FileLock(
                str(self._port_lock_file),
                timeout=float(os.environ.get("NVCF_PROXY_LOCK_TIMEOUT", str(LOCK_TIMEOUT))),
            )
            with lock:
                base_port = int(os.environ.get("NVCF_LOCAL_PROXY_PORT", str(DEFAULT_LOCAL_PROXY_PORT)))
                local_port = _find_available_port(base_port)
                self._proxy = NVCFLocalProxy(
                    api_key=self._api_key,
                    function_id=self._function_id,
                    local_port=local_port,
                )
                self._proxy.start()
                self.server_port = local_port
                self.chromium_port = local_port
                self.vlc_port = local_port
                self.vnc_port = local_port

            self._wait_for_vm_ready(timeout=int(os.environ.get("NVCF_READY_TIMEOUT", "300")))
            self._start_health_monitor()
            logger.info(
                "NVCF provider ready via local proxy on port %s (function=%s)",
                self.server_port,
                self._function_id,
            )
        except Exception:
            self.stop_emulator("")
            raise

    def get_ip_address(self, path_to_vm: str) -> str:
        del path_to_vm
        if not self.server_port:
            raise RuntimeError("NVCF provider not started - proxy port is missing")
        if self.vnc_port is None:
            raise RuntimeError("NVCF provider not started - VNC proxy port is missing")
        return (
            f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"
        )

    def save_state(self, path_to_vm: str, snapshot_name: str):
        del path_to_vm, snapshot_name
        raise NotImplementedError("Snapshots are not supported for NVCF provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        del snapshot_name
        self._stop_emulator(final=False)

    def _stop_emulator(self, final: bool):
        if final and self._final_cleanup_done:
            return

        self._stop_health_monitor()

        # Ensure deployer is available if we need to undeploy/delete a function we created.
        if (
            self._we_deployed_this
            and self._function_id
            and self._version_id
            and self._deployer is None
        ):
            try:
                self._ensure_deployer()
            except Exception as e:
                logger.warning("Failed to initialize NVCF deployer during cleanup: %s", e)

        if self._proxy is not None:
            try:
                self._proxy.stop()
            except Exception as e:
                logger.warning("Failed to stop local NVCF proxy: %s", e)
            finally:
                self._proxy = None

        should_cleanup_function = (
            self._we_deployed_this
            and self._deployer is not None
            and self._function_id is not None
            and self._version_id is not None
        )

        if should_cleanup_function:
            try:
                logger.info(
                    "Undeploying NVCF function %s (version %s)...",
                    self._function_id,
                    self._version_id,
                )
                self._deployer.undeploy(self._function_id, self._version_id)
            except Exception as e:
                logger.warning("Failed to undeploy NVCF function: %s", e)

            if final:
                try:
                    logger.info(
                        "Deleting NVCF function %s (version %s)...",
                        self._function_id,
                        self._version_id,
                    )
                    self._deployer.delete_function(self._function_id, self._version_id)
                except Exception as e:
                    logger.warning("Failed to delete NVCF function: %s", e)

        self.server_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.vnc_port = None

        if final:
            self._deployer = None
            self._we_deployed_this = False
            self._function_id = None
            self._version_id = None
            self._final_cleanup_done = True

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        del path_to_vm, region, args, kwargs
        self._stop_emulator(final=True)
