import base64
import io
import json
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import jsonlines
import pandas as pd
from PIL import Image

from openhands.core.logger import openhands_logger


def load_persona_dataset(persona_dataset_path: str, logger: openhands_logger) -> Tuple[List, List]:
    """
    Load the nemotron persona dataset from parquet files.
    """
    logger.debug(f"Loading persona dataset from {persona_dataset_path}...")

    parquet_filenames = list(Path(persona_dataset_path).glob('train-00000-of-00011.parquet'))

    persona_dfs, persona_df_weights = [], []
    total_records = 0

    for parquet_filename in parquet_filenames:
        df = pd.read_parquet(parquet_filename, memory_map=True)
        persona_dfs.append(df)
        persona_df_weights.append(len(df))
        total_records += len(df)

    logger.debug(f"  Loaded {total_records:,} total persona records from {len(parquet_filenames)} files.")

    return persona_dfs, persona_df_weights


def load_osworld_setup_list(osworld_setup_path: str, logger: openhands_logger) -> List[Optional[Dict]]:
    """
    Load the osworld setup dataset from json files.
    Returns a list with None at the first entry, indicating no setup.
    """
    osworld_setup_list = [None]
    with jsonlines.open(osworld_setup_path, 'r') as f:
        osworld_setup_list += list(f)

    logger.debug(f"  Loaded {len(osworld_setup_list)} total osworld setup records.")

    return osworld_setup_list


def load_example_instructions(example_instruction_path: str, logger: openhands_logger) -> List[str]:
    with open(example_instruction_path, 'r') as f:
        example_instructions = f.read().splitlines()

    logger.debug(f"  Loaded {len(example_instructions)} example instructions.")

    return example_instructions


def save_image(image_data: bytes | Image.Image, save_dir: Path | str, logger: Logger):
    if isinstance(image_data, bytes):
        with open(save_dir, 'wb') as f:
            f.write(image_data)
    elif isinstance(image_data, Image.Image):
        image_data.save(save_dir)
    else:
        raise NotImplementedError("Unsupported `image_data` type.")

    logger.debug(f"Image saved to {str(save_dir)}.")


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def bytes_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes to a data URI with image/png MIME type."""
    encoded_image_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded_image_str}"


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a data URI with image/png MIME type."""
    encoded_image_str = base64.b64encode(image_to_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{encoded_image_str}"
