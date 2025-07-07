from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set
import pandas as pd


DEFAULT_NORMAL_PREFIX = os.getenv("EVAL_DOCKER_IMAGE_PREFIX", "xingyaoww/").rstrip("/")
OFFICIAL_MM_PREFIX = "swebench"

def instance_id_to_image(
    instance_id: str,
    *,
    prefix: str | None = None,
    multimodal: bool = False,
) -> str:
    """Return full Docker image name for a given instance_id."""
    _prefix = (
        (prefix or "").rstrip("/") if prefix is not None else (OFFICIAL_MM_PREFIX if multimodal else DEFAULT_NORMAL_PREFIX)
    )

    repo, issue = instance_id.split("__") if "__" in instance_id else (None, None)

    if _prefix in {OFFICIAL_MM_PREFIX, "swebench"}:
        if repo and issue:
            core = f"sweb.eval.x86_64.{repo}_1776_{issue}"
        else:
            core = f"sweb.eval.x86_64.{instance_id}"
        return f"{_prefix}/{core}:latest".lower()

    suffix = instance_id.replace("__", "_s_").lower()
    core = f"sweb.eval.x86_64.{suffix}"
    return f"{_prefix}/{core}".lower()

def parquet_to_instance_ids(parquet_path: Path) -> Iterable[str]:
    """Yield instance_id strings from a parquet file."""
    if not parquet_path.exists():
        print(f"[WARN] {parquet_path} not found – skipping.")
        return
    try:
        df = pd.read_parquet(parquet_path, columns=["instance"])
        for inst in df["instance"]:
            if isinstance(inst, dict):
                instance_id = inst.get("instance_id")
                if instance_id is not None:
                    yield instance_id
            else:
                try:
                    d = json.loads(inst)
                    instance_id = d.get("instance_id")
                    if instance_id is not None:
                        yield instance_id
                except Exception:
                    continue
    except Exception:
        try:
            df = pd.read_parquet(parquet_path, columns=["instance_id"])
            for iid in df["instance_id"]:
                yield iid
        except Exception as e:
            print(f"[ERROR] Unable to read instance ids from {parquet_path}: {e}")


def gather_images(
    data_dir: Path,
    *,
    multimodal: bool = False,
    prefix: str | None = None,
) -> Set[str]:
    """Collect image names required by a dataset dir."""
    multimodal = "multimodal" in str(data_dir).lower()

    images: set[str] = set()
    for fname in ["train.parquet", "validation.parquet", "val.parquet"]:
        for iid in parquet_to_instance_ids(data_dir / fname):
            if iid:
                images.add(instance_id_to_image(iid, multimodal=multimodal, prefix=prefix))
    return images

def sanitize_image_name(image: str) -> str:
    """Make image name filesystem-safe."""
    return image.replace("/", "_").replace(":", "_")

def _run(cmd: List[str], cwd: Path | None = None, *, clean_bind_env: bool = False) -> None:
    """Run shell command and raise on failure."""
    env = os.environ.copy()
    if clean_bind_env:
        for var in ["SINGULARITY_BINDPATH", "APPTAINER_BINDPATH"]:
            env.pop(var, None)
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}") from e

def build_sif_for_image(image: str, dest_dir: Path, temp_base: Path) -> None:
    sanitized_name = sanitize_image_name(image)
    build_dir = temp_base / sanitized_name
    sif_path = dest_dir / f"{sanitized_name}.sif"

    print(f"[BUILD] {image} -> {sif_path}")

    if sif_path.exists():
        print("[SKIP] Exists")
        return

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Generating Singularity build context…")
    _run(
        [
            "python",
            "-m",
            "openhands.runtime.utils.singularity_runtime_build",
            "--base_image",
            image,
            "--build_folder",
            str(build_dir),
            "--force_rebuild",
            "--platform",
            "linux/amd64",
        ]
    )

    def_path = build_dir / "singularity.def"
    if def_path.exists():
        lines = def_path.read_text().splitlines()
        if not any("mkdir -p /workspace" in l for l in lines):
            patched: list[str] = []
            inserted = False
            for line in lines:
                patched.append(line)
                if not inserted and line.strip() == "%post":
                    patched.append("    mkdir -p /workspace  # auto-inserted by collect_and_pull_images.py")
                    inserted = True
            def_path.write_text("\n".join(patched) + "\n")

    print("[STEP 2] Building .sif image with Singularity…")
    _run([
        "singularity",
        "build",
        str(sif_path.resolve()),
        "singularity.def",
    ], cwd=build_dir, clean_bind_env=True)

    if image.startswith("swebench/"):
        new_sif_path = sif_path.with_name(f"docker.io_{sif_path.name}")
        if not new_sif_path.exists():
            os.rename(sif_path, new_sif_path)
            sif_path = new_sif_path

    print(f"[OK]   {sif_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Docker images from a parquet file and build Singularity SIFs.")
    parser.add_argument("--parquet-file", type=Path, required=True, help="Path to a SWE-Bench parquet file (train/validation etc.)")
    parser.add_argument("--prefix", type=str, default=None, help="Override Docker image namespace prefix")
    parser.add_argument("--dest-dir", type=Path, default=None, help="Directory to store .sif images (default: <workspace>/singularity_images)")
    parser.add_argument("--temp-base", type=Path, default=None, help="Base directory for temporary build folders (default: <dest>/temp_dif)")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index of first image to process (after sorting)")
    parser.add_argument("--end-index", type=int, default=None, help="1-based index of last image to process (inclusive). If omitted, process till end.")
    parser.add_argument("--log-name", type=str, default=None, help="Capture combined stdout/stderr to a log file under images_process/log/")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent

    script_dir = Path(__file__).resolve().parent
    cache_base = script_dir / "_singularity_cache"
    for env_var in [
        "APPTAINER_TMPDIR",
        "SINGULARITY_TMPDIR",
        "APPTAINER_CACHEDIR",
        "APPTAINER_LOCALCACHEDIR",
    ]:
        if not os.getenv(env_var):
            local_path = cache_base / env_var.lower()
            try:
                local_path.mkdir(parents=True, exist_ok=True)
                os.environ[env_var] = str(local_path)
            except Exception as exc:
                print(f"[WARN] Cache init failed for {env_var}: {exc}")

    dest_dir = Path(os.getenv("DEST_DIR", str(args.dest_dir or (workspace_root / "singularity_images"))))
    temp_base = Path(os.getenv("TEMP_BASE", str(args.temp_base or (dest_dir / "temp_dif"))))

    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_base.mkdir(parents=True, exist_ok=True)

    if args.log_name:
        script_dir = Path(__file__).resolve().parent
        fixed_log_dir = script_dir / "log"
        fixed_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = fixed_log_dir / args.log_name
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".txt")

        class _Tee:
            """Simple tee writing to multiple streams."""

            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)

            def flush(self):
                for s in self.streams:
                    s.flush()

        log_fh = log_path.open("a", buffering=1)
        sys.stdout = _Tee(sys.stdout, log_fh)  # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, log_fh)  # type: ignore[assignment]
        print(f"[LOG] Writing combined stdout/stderr to {log_path}")

    if not args.parquet_file.exists():
        sys.exit(f"[ERROR] Parquet file {args.parquet_file} not found")

    multimodal_detect = "multimodal" in str(args.parquet_file).lower()
    images = sorted({
        instance_id_to_image(iid, multimodal=multimodal_detect, prefix=args.prefix)
        for iid in parquet_to_instance_ids(args.parquet_file)
        if iid
    })
    total_images = len(images)
    if total_images == 0:
        sys.exit("[ERROR] No images found – aborting.")

    print(f"[INFO] {total_images} images detected")

    start_idx = max(1, args.start_index)
    end_idx = args.end_index or total_images
    if start_idx > total_images:
        sys.exit(f"[ERROR] start-index ({start_idx}) exceeds number of images ({total_images})")
    end_idx = min(end_idx, total_images)

    print(f"[INFO] Processing {start_idx}-{end_idx}")
    selected_images = images[start_idx - 1 : end_idx]

    processed = 0
    for img in selected_images:
        try:
            build_sif_for_image(img, dest_dir, temp_base)
            processed += 1
        except Exception as exc:
            print(f"[ERROR] Failed processing {img}: {exc}")

    print(f"[DONE] Built {processed}/{len(selected_images)} images")


if __name__ == "__main__":
    main()

# python pull_swe_images.py --parquet-file /lustre/fsw/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/80-data/train.parquet --start-index 1 --end-index 2 --dest-dir /lustre/fsw/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/images_process/test --log-name log_test
