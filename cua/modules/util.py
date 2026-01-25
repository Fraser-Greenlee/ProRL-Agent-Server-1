import base64
import io
import re
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict
import json

import ipdb
import jsonlines
import pandas as pd
from PIL import Image
from qwen_vl_utils import fetch_image

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


def save_image(image_data: bytes | Image.Image, save_dir: Path | str, logger: Logger):
    """
    Save image_data to save_dir.
    """
    if isinstance(image_data, bytes):
        with open(save_dir, 'wb') as f:
            f.write(image_data)
    elif isinstance(image_data, Image.Image):
        image_data.save(save_dir)
    else:
        raise NotImplementedError("Unsupported `image_data` type.")

    logger.debug(f"✓ Image saved to {str(save_dir)}.")


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def image_to_base64(image: Image.Image) -> str:
    encoded_image_str = base64.b64encode(image_to_bytes(image)).decode("utf-8")
    return f"data:image;base64,{encoded_image_str}"


def process_image(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    if min_pixels != -1 and max_pixels != -1:
        image_dict = {
            "image": image,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
    else:
        image_dict = {
            "image": image,
        }
    return fetch_image(image_dict)


def build_messages(line: Dict, min_pixels: int, max_pixels: int) -> List:
    messages: List[Dict] = line.pop("prompt")  # expected to be [{"role": "user", "content": "..."}, ...]

    if "images" in line and line["images"] is not None:
        images: List[Image.Image] = [process_image(image, min_pixels, max_pixels) for image in line.pop("images")]

        image_idx = 0
        for message in messages:
            content = message["content"]
            content_list = []
            for segment in re.split("(<image>)", content):
                if segment == "<image>":
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(images[image_idx])}
                    })
                    image_idx += 1
                elif segment == "":
                    continue  # empty string should be omitted
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        if image_idx != len(images):
            ipdb.set_trace()  # we haven't used all images in the messages
            pass

    return messages
