import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import json

def merge_multiple_dataframes(dataframes):
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    return merged_df

def pre_process_r2egym_instance(r2egym_instance):
    r2egym_instance = pd.Series(r2egym_instance)
    r2egym_instance['data_source'] = 'swebench'  # keep using swebench handler logic
    r2egym_instance['instance_id'] = (
        r2egym_instance['docker_image'].replace('/', '_').replace(':', '_')
    )
    docker_image = r2egym_instance['docker_image']
    if ':' in docker_image:
        repo_part, version_part = docker_image.split(':', 1)
    else:
        repo_part, version_part = docker_image, 'latest'
    r2egym_instance['repo'] = repo_part
    r2egym_instance['version'] = version_part

    if ('base_commit' not in r2egym_instance) or pd.isna(r2egym_instance['base_commit']):
        if 'commit_hash' in r2egym_instance and not pd.isna(r2egym_instance['commit_hash']):
            r2egym_instance['base_commit'] = r2egym_instance['commit_hash']
        else:
            r2egym_instance['base_commit'] = version_part
    r2egym_instance['data_split'] = 'r2egym'
    r2egym_instance['data_source'] = 'swebench'
    return r2egym_instance

def pre_process_swebench_instance(swebench_instance):
    payload = swebench_instance.get("instance") if isinstance(swebench_instance, pd.Series) else swebench_instance["instance"]
    payload["data_split"] = "swebench"
    payload['data_source'] = 'swebench'

    return pd.Series(payload)

def pre_process_swebench_mm_instance(swebench_mm_instance):
    swebench_mm_instance['data_split'] = 'swebench_multimodal'
    swebench_mm_instance['data_source'] = 'swebench'
    return swebench_mm_instance

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data: merge and pre-process parquet files from a directory (SWE-Bench, R2E-Gym, etc.).")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing parquet files to process.")

    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    print("-------------data_dir-----------------")
    print(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"[ERROR] --data-dir {data_dir} is not a valid directory")
    parquet_paths = sorted(p.resolve() for p in data_dir.rglob("*.parquet"))
    print(parquet_paths)
    print(f"[INFO] Found {len(parquet_paths)} parquet files:")
    for p in parquet_paths:
        print("  ", p)

    final_list = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        # r2egyn
        if "docker_image" in df.columns:
            df = df.apply(pre_process_r2egym_instance, axis=1)
            final_list.append(df)
        # swebench_mm
        elif "image_assets" in df.columns:
            df = df.apply(pre_process_swebench_mm_instance, axis=1)
            final_list.append(df)
        # swebench
        else:
            df = df.apply(pre_process_swebench_instance, axis=1)
            final_list.append(df)

    final_df = merge_multiple_dataframes(final_list)

    print(final_df.head())
    def _normalise(v):
        if isinstance(v, np.ndarray):
            return v.tolist()          # or json.dumps(v.tolist())
    final_df['FAIL_TO_PASS'] = final_df['FAIL_TO_PASS'].apply(_normalise)
    final_df['PASS_TO_PASS'] = final_df['PASS_TO_PASS'].apply(_normalise)
    print(final_df.iloc[0])
    # final_df.to_parquet(data_dir / "merged2.parquet")


if __name__ == "__main__":
    main()
