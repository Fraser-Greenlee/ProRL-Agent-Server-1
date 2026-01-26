import io
import random
from argparse import Namespace
from logging import Logger
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from cua.modules.util import image_to_bytes, bytes_to_image
from openhands.core.logger import openhands_logger


logger = openhands_logger.getChild('openai_wrapper')


class ParserController:
    """
    Wrapper class that supports all interaction with remote screen-parser server.
    """
    def __init__(self, args: Namespace):
        self.server_url = f"http://{args.parser_node}:8000/parse"

    def parse_screenshot(self, image_bytes: bytes) -> Tuple:
        """
        Parse the screenshot and return the parsed image.
        Returns tuple of image with bboxes (bytes) and the bbox list
        """

        # Send POST request
        response = requests.post(self.server_url, files={"file": ("", image_bytes, "image/png")}, timeout=10)

        if response.status_code == 200:
            result = response.json()

            # for ease of debugging, sort bboxes based on their x-coordinate
            result["boxes"] = sorted(result["boxes"], key=lambda box: box[0])

        else:
            logger.debug(f"[parse_screenshot] ❌ Server Error {response.status_code}: {response.text}")
            raise ConnectionError

        parsed_image_bytes = self.prepare_image_with_som(image_bytes, result["boxes"])

        return parsed_image_bytes, result["boxes"]

    @staticmethod
    def prepare_image_with_som(image: Image.Image | bytes, bbox_list: List, label_on_image: bool = True) -> bytes:
        """
        Draws bounding boxes and indices with tight margins.
        Returns new image with bboxes (bytes) and the sorted bbox list.
        """
        if isinstance(image, bytes):
            image_source = bytes_to_image(image)
        else:
            image_source = image.convert("RGB")

        annotated_image = image_source.copy()
        draw = ImageDraw.Draw(annotated_image)
        width, height = annotated_image.size

        # --- Configuration ---
        box_color = (0, 0, 139)  # Dark Blue
        label_bg_color = (0, 0, 139)  # Dark Blue
        text_color = (255, 255, 255)  # White
        box_width = 2

        # Load Font
        try:
            # You might need to adjust the path or use a default font depending on your OS
            font = ImageFont.truetype("LiberationSans-Bold.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # --- PASS 1: Draw All Rectangles (Standard Order) ---
        # We draw outlines first so they never overlap any text
        for box in bbox_list:
            # Ensure coordinates are integers
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

        # --- PREPARE LAYERING LOGIC ---
        if label_on_image:
            # Sort indices based on AREA (Largest -> Smallest) so small labels sit on top
            sorted_indices = sorted(
                range(len(bbox_list)),
                key=lambda i: (bbox_list[i][2] - bbox_list[i][0]) * (bbox_list[i][3] - bbox_list[i][1]),
                reverse=True
            )

            # --- PASS 2: Draw All Labels (Sorted Order) ---
            for i in sorted_indices:
                box = bbox_list[i]
                x1, y1, x2, y2 = map(int, box)
                label_text = str(i)

                # Calculate text dimensions
                if hasattr(font, "getbbox"):
                    text_bbox = font.getbbox(label_text)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                else:
                    # Fallback for older PIL versions
                    text_w, text_h = draw.textsize(label_text, font=font)

                # --- Position Logic (Tightened) ---
                label_x = x1
                label_y = y2 + 1

                # Boundary Check: Bottom of Image (Flip Up)
                if label_y + text_h + 2 > height:
                    label_y = y2 - text_h - 3

                # Boundary Check: Right of Image (Shift Left)
                if label_x + text_w + 2 > width:
                    label_x = width - text_w - 2

                # Draw Background Box
                draw.rectangle(
                    [label_x, label_y, label_x + text_w + 2, label_y + text_h + 2],
                    fill=label_bg_color,
                    outline=box_color,
                    width=1
                )

                # Draw Text
                draw.text(
                    (label_x + 1, label_y),
                    label_text,
                    fill=text_color,
                    font=font
                )

        # Convert PIL back to bytes
        return image_to_bytes(annotated_image)



