from pathlib import Path
from typing import Tuple, List, Dict
import json

import jsonlines
import pandas as pd

from openhands.core.logger import openhands_logger


def load_persona_dataset(persona_dataset_path: str, logger: openhands_logger) -> Tuple[List, List]:
    """
    Load the nemotron persona dataset from parquet files.
    """
    logger.debug(f"Loading persona dataset from {persona_dataset_path}...")

    # Load all parquet files
    parquet_filenames = list(Path(persona_dataset_path).glob('train-00000-of-00011.parquet'))  # for now, just retain 1 parquet

    persona_dfs, persona_df_weights = [], []
    total_records = 0

    for parquet_filename in parquet_filenames:
        df = pd.read_parquet(parquet_filename, memory_map=True)
        persona_dfs.append(df)
        persona_df_weights.append(len(df))
        total_records += len(df)

    logger.debug(f"  ✓ Loaded {total_records:,} total persona records from {len(parquet_filenames)} files.")

    return persona_dfs, persona_df_weights


def load_osworld_setup_list(osworld_setup_path: str, logger: openhands_logger) -> List[None | Dict]:
    """
    Load the osworld setup dataset from json files.
    Returns a list with None at the first entry, indicating no setup.
    """
    osworld_setup_list = [None]
    with jsonlines.open(osworld_setup_path, 'r') as f:
        osworld_setup_list += list(f)

    logger.debug("  ✓ Loaded {len(self.osworld_setup_dataset)} total osworld setup records")

    return osworld_setup_list

