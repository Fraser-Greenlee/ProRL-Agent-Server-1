#!/usr/bin/env python3
"""
Cleanup script for NVCF functions.

Behavior:
  1) Delete all non-ACTIVE function versions (no name-prefix filter).
  2) For ACTIVE deployments, undeploy+delete only those matching name prefix.
  3) Print remaining ACTIVE deployment count and names.

Required environment variables:
  - NGC_API_KEY
  - NGC_ORG (or OGC_ORG as fallback)

Optional environment variables:
  - NGC_TEAM (defaults to "no-team")
"""

from __future__ import annotations

import argparse
from collections import Counter
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ngcsdk import Client


logger = logging.getLogger("cleanup_nvcf_active_functions")


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        return {}


def _extract_list(payload: Any, keys: Iterable[str]) -> List[Any]:
    if isinstance(payload, list):
        return payload
    payload_dict = _to_dict(payload)
    for key in keys:
        value = payload_dict.get(key)
        if isinstance(value, list):
            return value
    for key in keys:
        value = getattr(payload, key, None)
        if isinstance(value, list):
            return value
    return []


def _record_get(record: Any, keys: Iterable[str], default: Any = None) -> Any:
    rec_dict = _to_dict(record)
    for key in keys:
        if key in rec_dict and rec_dict[key] is not None:
            return rec_dict[key]
    for key in keys:
        value = getattr(record, key, None)
        if value is not None:
            return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete non-ACTIVE function versions, and undeploy+delete ACTIVE ones "
            "matching name prefix."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be undeployed/deleted.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log verbosity.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=180,
        help="Seconds to wait for undeploy transition before delete attempt.",
    )
    parser.add_argument(
        "--wait-interval",
        type=int,
        default=5,
        help="Polling interval (seconds) while waiting after undeploy.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="osworld-nvcf",
        help=(
            "Only cleanup ACTIVE deployments whose function name starts with this prefix. "
            "Set to empty string to disable name filtering."
        ),
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s %(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def _build_client() -> Client:
    api_key = os.environ.get("NGC_API_KEY")
    org_name = os.environ.get("NGC_ORG") or os.environ.get("OGC_ORG")
    team_name = os.environ.get("NGC_TEAM", "no-team")

    if not api_key:
        raise ValueError("Missing NGC_API_KEY in environment.")
    if not org_name:
        raise ValueError("Missing NGC_ORG (or OGC_ORG fallback) in environment.")

    client = Client()
    client.configure(api_key, org_name=org_name, team_name=team_name)
    logger.info("Configured NVCF client for org=%s team=%s", org_name, team_name)
    return client


def _is_active_status(status: str) -> bool:
    return str(status or "").upper() == "ACTIVE"


def _resolve_function_name(
    client: Client,
    function_id: str,
    version_id: str,
    deployment_record: Any,
) -> str:
    name = _record_get(
        deployment_record,
        ["functionName", "function_name", "name"],
        default="",
    )
    if name:
        return str(name)
    try:
        info = client.cloud_function.functions.info(
            function_id=function_id,
            function_version_id=version_id,
        )
        function_block = _record_get(info, ["function"], default={})
        name = _record_get(function_block, ["name"], default=_record_get(info, ["name"], default=""))
        return str(name or "")
    except Exception:
        return ""


def _discover_all_function_versions(
    client: Client,
) -> List[Tuple[str, str, str, str, bool]]:
    """Return unique (function_id, version_id, function_name, status, owned_by_different_account)."""
    functions_payload = client.cloud_function.functions.list()
    functions = _extract_list(functions_payload, ["functions", "data", "items"])

    records: Dict[Tuple[str, str], Tuple[str, str, bool]] = {}
    for fn in functions:
        function_id = _record_get(fn, ["id", "functionId", "function_id"])
        version_id = _record_get(fn, ["versionId", "functionVersionId", "function_version_id"])
        if not function_id or not version_id:
            continue

        name = str(_record_get(fn, ["name", "functionName", "function_name"], default="") or "")
        status = str(_record_get(fn, ["status", "functionStatus", "state"], default="UNKNOWN") or "UNKNOWN").upper()
        owned_by_different_account = _as_bool(
            _record_get(
                fn,
                ["ownedByDifferentAccount", "owned_by_different_account"],
                default=False,
            )
        )
        records[(str(function_id), str(version_id))] = (name, status, owned_by_different_account)

    return sorted(
        [
            (fid, vid, name, status, owned_by_different_account)
            for (fid, vid), (name, status, owned_by_different_account) in records.items()
        ],
        key=lambda x: (x[2], x[0], x[1]),
    )


def _discover_active_deployments(
    client: Client,
    name_prefix: str,
) -> List[Tuple[str, str, str]]:
    """Return unique (function_id, version_id, function_name) for ACTIVE deployments."""
    deployments_payload = client.cloud_function.functions.deployments.list()
    deployments = _extract_list(deployments_payload, ["deployments", "data", "items"])

    active_records: Dict[Tuple[str, str], str] = {}
    for dep in deployments:
        status = str(_record_get(dep, ["functionStatus", "status", "deploymentStatus", "state"], "")).upper()
        if not _is_active_status(status):
            continue

        function_id = _record_get(dep, ["functionId", "function_id", "id"])
        version_id = _record_get(dep, ["functionVersionId", "function_version_id", "versionId", "version_id"])
        if not function_id or not version_id:
            continue

        function_id = str(function_id)
        version_id = str(version_id)
        function_name = _resolve_function_name(client, function_id, version_id, dep)
        if name_prefix and not function_name.startswith(name_prefix):
            logger.debug(
                "Skipping ACTIVE function %s (%s:%s) due to name-prefix filter '%s'.",
                function_name or "<unknown>",
                function_id,
                version_id,
                name_prefix,
            )
            continue
        active_records[(function_id, version_id)] = function_name

    return sorted(
        [(fid, vid, fname) for (fid, vid), fname in active_records.items()],
        key=lambda x: (x[2], x[0], x[1]),
    )


def _is_owned_by_different_account(
    client: Client,
    function_id: str,
    version_id: str,
    ownership_cache: Dict[Tuple[str, str], bool],
) -> bool:
    key = (function_id, version_id)
    if key in ownership_cache:
        return ownership_cache[key]

    owned_by_different_account = False
    try:
        info = client.cloud_function.functions.info(
            function_id=function_id,
            function_version_id=version_id,
        )
        function_block = _record_get(info, ["function"], default={})
        owned_by_different_account = _as_bool(
            _record_get(
                function_block,
                ["ownedByDifferentAccount", "owned_by_different_account"],
                default=_record_get(
                    info,
                    ["ownedByDifferentAccount", "owned_by_different_account"],
                    default=False,
                ),
            )
        )
    except Exception:
        # If ownership lookup fails, treat as deletable and let API decide.
        owned_by_different_account = False

    ownership_cache[key] = owned_by_different_account
    return owned_by_different_account


def _wait_until_not_active(
    client: Client,
    function_id: str,
    version_id: str,
    timeout: int,
    interval: int,
) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            info = client.cloud_function.functions.info(
                function_id=function_id,
                function_version_id=version_id,
            )
            status = str(
                _record_get(
                    _record_get(info, ["function"], default={}),
                    ["status"],
                    default=_record_get(info, ["status"], default="UNKNOWN"),
                )
            ).upper()
            if "ACTIVE" not in status:
                return
        except Exception:
            # If info call fails, continue to delete attempt.
            return
        time.sleep(interval)


def _cleanup_pair(
    client: Client,
    function_id: str,
    version_id: str,
    function_name: str,
    dry_run: bool,
    wait_timeout: int,
    wait_interval: int,
) -> bool:
    logger.info(
        "Target function name=%s function_id=%s version_id=%s",
        function_name or "<unknown>",
        function_id,
        version_id,
    )
    if dry_run:
        logger.info("Dry run: would undeploy + delete this function version.")
        return True

    ok = True
    undeploy_ok = True
    try:
        client.cloud_function.functions.deployments.delete(
            function_id=function_id,
            function_version_id=version_id,
            graceful=True,
        )
        logger.info("Undeploy requested.")
    except Exception as e:
        logger.error("Undeploy failed: %s", e)
        undeploy_ok = False
        ok = False

    try:
        client.cloud_function.functions.delete(
            function_id=function_id,
            function_version_id=version_id,
        )
        logger.info("Delete requested (immediate).")
    except Exception as e:
        logger.warning("Immediate delete failed: %s", e)
        # Fallback: only wait if undeploy call succeeded, then retry delete once.
        if undeploy_ok and wait_timeout > 0:
            logger.info(
                "Retrying delete after waiting up to %ss (interval=%ss).",
                wait_timeout,
                wait_interval,
            )
            _wait_until_not_active(
                client=client,
                function_id=function_id,
                version_id=version_id,
                timeout=wait_timeout,
                interval=wait_interval,
            )
            try:
                client.cloud_function.functions.delete(
                    function_id=function_id,
                    function_version_id=version_id,
                )
                logger.info("Delete requested (after wait/retry).")
            except Exception as e2:
                logger.error("Delete failed after wait/retry: %s", e2)
                ok = False
        else:
            logger.error("Delete failed: %s", e)
            ok = False

    return ok


def _delete_function_only(
    client: Client,
    function_id: str,
    version_id: str,
    function_name: str,
    dry_run: bool,
) -> bool:
    logger.info(
        "Delete non-active function name=%s function_id=%s version_id=%s",
        function_name or "<unknown>",
        function_id,
        version_id,
    )
    if dry_run:
        logger.info("Dry run: would delete this non-active function version.")
        return True
    try:
        client.cloud_function.functions.delete(
            function_id=function_id,
            function_version_id=version_id,
        )
        logger.info("Delete requested.")
        return True
    except Exception as e:
        logger.error("Delete failed: %s", e)
        return False


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)

    try:
        client = _build_client()
    except Exception as e:
        logger.error("Failed to configure client: %s", e)
        return 2

    prefix = args.name_prefix.strip()

    try:
        all_functions = _discover_all_function_versions(client)
    except Exception as e:
        logger.error("Failed to list function versions: %s", e)
        return 3

    non_active_functions = [
        (fid, vid, name, status, owned_by_different_account)
        for fid, vid, name, status, owned_by_different_account in all_functions
        if not _is_active_status(status)
    ]
    status_counts = Counter(status for _fid, _vid, _name, status, _owned in all_functions)
    logger.info("Total function versions discovered: %d", len(all_functions))
    logger.info("Function status distribution: %s", dict(status_counts))
    logger.info(
        "Found %d non-active function version(s) to delete (no prefix filter).",
        len(non_active_functions),
    )

    success = 0
    total_operations = 0
    skipped_owned_non_active = 0
    for function_id, version_id, function_name, _status, owned_by_different_account in non_active_functions:
        if owned_by_different_account:
            skipped_owned_non_active += 1
            logger.info(
                "Skipping non-active function not owned by this account: name=%s function_id=%s version_id=%s",
                function_name or "<unknown>",
                function_id,
                version_id,
            )
            continue
        total_operations += 1
        if _delete_function_only(
            client=client,
            function_id=function_id,
            version_id=version_id,
            function_name=function_name,
            dry_run=args.dry_run,
        ):
            success += 1
    if skipped_owned_non_active:
        logger.info(
            "Skipped %d non-active function version(s) owned by different account.",
            skipped_owned_non_active,
        )

    try:
        active_targets = _discover_active_deployments(client, prefix)
    except Exception as e:
        logger.error("Failed to discover ACTIVE deployments: %s", e)
        return 3

    if prefix:
        logger.info(
            "Found %d ACTIVE deployment(s) matching prefix '%s' for undeploy+delete.",
            len(active_targets),
            prefix,
        )
    else:
        logger.info(
            "Found %d ACTIVE deployment(s) for undeploy+delete (no prefix filter).",
            len(active_targets),
        )

    ownership_cache: Dict[Tuple[str, str], bool] = {}
    skipped_owned_active = 0
    for function_id, version_id, function_name in active_targets:
        if _is_owned_by_different_account(client, function_id, version_id, ownership_cache):
            skipped_owned_active += 1
            logger.info(
                "Skipping ACTIVE function not owned by this account: name=%s function_id=%s version_id=%s",
                function_name or "<unknown>",
                function_id,
                version_id,
            )
            continue
        total_operations += 1
        if _cleanup_pair(
            client=client,
            function_id=function_id,
            version_id=version_id,
            function_name=function_name,
            dry_run=args.dry_run,
            wait_timeout=args.wait_timeout,
            wait_interval=args.wait_interval,
        ):
            success += 1
    if skipped_owned_active:
        logger.info(
            "Skipped %d ACTIVE function version(s) owned by different account.",
            skipped_owned_active,
        )

    try:
        remaining_active = _discover_active_deployments(client, "")
    except Exception as e:
        logger.error("Failed to list remaining ACTIVE deployments: %s", e)
        remaining_active = []

    remaining_names = sorted({name or "<unknown>" for _, _, name in remaining_active})
    logger.info("ACTIVE deployments left: %d", len(remaining_active))
    logger.info("ACTIVE function names left (%d): %s", len(remaining_names), ", ".join(remaining_names))

    logger.info("Cleanup finished: %d/%d operations succeeded.", success, total_operations)
    return 0 if success == total_operations else 1


if __name__ == "__main__":
    raise SystemExit(main())
