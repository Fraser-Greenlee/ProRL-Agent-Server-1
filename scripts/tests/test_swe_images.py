import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openhands.nvidia.swe_agent.utils import evaluate_agent as _evaluate_agent


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ground-truth patches from the SWE-bench dataset using the OpenHands evaluation harness (no agent run).',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/80-data/train.parquet',
        help="Path to a parquet file that contains the SWE-bench dataset with an 'instance' column.",
    )
    parser.add_argument(
        '--output',
        type=str,
        default='swebench_gold_eval_results.jsonl',
        help='Output file to write evaluation results (JSONL).',
    )
    parser.add_argument(
        '--num-instances',
        type=int,
        default=None,
        help='Number of instances to evaluate (default: 1).',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=64,
        help='Maximum number of concurrent evaluations to run (default: 64, matching run_swebench.py).',
    )
    parser.add_argument(
        '--allow-skip-eval',
        action='store_true',
        help='Skip evaluation when the git_patch is empty or None (mirrors utils._evaluate_agent default).',
    )
    return parser.parse_args()


async def _evaluate_instances(
    instances: list[dict], concurrency: int, allow_skip: bool
):
    semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_single(idx: int, inst: dict):
        async with semaphore:
            try:
                patch = inst.get('patch')
                if patch is None:
                    return {
                        'instance_id': inst.get('instance_id', idx),
                        'trajectory_id': inst.get('trajectory_id', idx),
                        'resolved': False,
                        'error': 'No patch provided in instance',
                    }
                rep = await _evaluate_agent(
                    patch, pd.Series(inst), sid=f'gold_{idx}', allow_skip=allow_skip
                )

                resolved_flag = False
                if isinstance(rep, dict):
                    if 'report' in rep and isinstance(rep['report'], dict):
                        resolved_flag = rep['report'].get('resolved', False)
                    elif 'resolved' in rep:
                        resolved_flag = rep.get('resolved', False)
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': resolved_flag,
                    'evaluation': rep,
                }
            except Exception as e:
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': False,
                    'error': str(e),
                }

    tasks = [_evaluate_single(i, inst) for i, inst in enumerate(instances)]
    return await asyncio.gather(*tasks)


def main():
    args = _parse_args()

    dataset_df = pd.read_parquet(args.dataset_path)
    if 'instance' not in dataset_df.columns:
        raise ValueError("The provided parquet file must contain an 'instance' column.")

    if args.num_instances is not None:
        dataset_df = dataset_df.head(args.num_instances)
    len(dataset_df)

    instances: list[dict] = []
    for idx, row in dataset_df.iterrows():
        inst_series = pd.Series(row['instance'])
        inst_series = inst_series.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        if 'trajectory_id' not in inst_series or pd.isna(
            inst_series.get('trajectory_id')
        ):
            inst_series['trajectory_id'] = idx
        instances.append(inst_series.to_dict())

    start_ts = time.time()
    results = asyncio.run(
        _evaluate_instances(
            instances, concurrency=args.concurrency, allow_skip=args.allow_skip_eval
        )
    )
    time.time() - start_ts

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as fp:
        for res in results:
            json.dump(res, fp)
            fp.write('\n')
    print(f'[evaluate_gold] Results written to {output_path}')

    resolved_cnt = sum(1 for r in results if r.get('resolved'))
    print(
        f'[evaluate_gold] Resolved {resolved_cnt}/{len(results)} instances ({resolved_cnt / len(results):.2%})'
    )


if __name__ == '__main__':
    main()
