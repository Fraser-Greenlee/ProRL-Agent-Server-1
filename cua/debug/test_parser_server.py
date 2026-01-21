import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import ipdb
import requests


def test_parsing(server_url: str, image_path: str):
    print(f"--- Sending request to {server_url} ---")

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Load image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    start_time = time.time()

    try:
        # Send POST request
        response = requests.post(server_url, files={"file": (image_path, image_bytes, "image/png")}, timeout=10)

        if response.status_code == 200:
            result = response.json()
            latency = (time.time() - start_time) * 1000

            # Print Summary
            print(f"✅ Success! (Latency: {latency:.2f}ms)")
            print(f"Result: {result}")

        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            result = None

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Failed.")
        result = None

    except Exception as e:
        print(f"❌ Error: {e}")
        result = None

    return result


def draw_bboxes(image: Image.Image, bbox_list: list) -> Image.Image:
    """
    Draws bounding boxes and indices with tight margins.
    Style: Blue Boxes. Blue Label Background. White Text.
    Layering: Labels for SMALLER boxes are drawn ON TOP of larger boxes.
    """
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size

    # --- Configuration ---
    BOX_COLOR = (0, 0, 139)       # Dark Blue
    LABEL_BG_COLOR = (0, 0, 139)  # Dark Blue
    TEXT_COLOR = (255, 255, 255)  # White

    BOX_WIDTH = 2

    # Load Font
    try:
        font = ImageFont.truetype("LiberationSans-Bold.ttf", 11)
    except IOError:
        font = ImageFont.load_default()

    # --- PASS 1: Draw All Rectangles (Standard Order) ---
    # We draw outlines first so they never overlap any text
    for box in bbox_list:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)

    # --- PREPARE LAYERING LOGIC ---
    # 1. Create a list of indices: [0, 1, 2, ... len(bbox_list)]
    # 2. Sort these indices based on the AREA of their corresponding box (Largest -> Smallest)
    #    (x2 - x1) * (y2 - y1) is the area
    sorted_indices = sorted(
        range(len(bbox_list)),
        key=lambda i: (bbox_list[i][2] - bbox_list[i][0]) * (bbox_list[i][3] - bbox_list[i][1]),
        reverse=True
    )

    # --- PASS 2: Draw All Labels (Sorted Order) ---
    # We iterate through the sorted indices.
    # Large boxes come first (bottom layer). Small boxes come last (top layer).
    for i in sorted_indices:
        box = bbox_list[i]
        x1, y1, x2, y2 = box
        label_text = str(i) # We use 'i' to preserve the original index ID

        # Calculate text dimensions
        text_bbox = font.getbbox(label_text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # --- Position Logic (Tightened) ---
        label_x = x1
        label_y = y2 + 1 # Anchor bottom-left

        # Boundary Check: Bottom of Image (Flip Up)
        if label_y + text_h + 2 > height:
            label_y = y2 - text_h - 3

        # Boundary Check: Right of Image (Shift Left)
        if label_x + text_w + 2 > width:
            label_x = width - text_w - 2

        # Draw Background Box
        draw.rectangle(
            [label_x, label_y, label_x + text_w + 2, label_y + text_h + 2],
            fill=LABEL_BG_COLOR,
            outline=BOX_COLOR,
            width=1
        )

        # Draw Text
        draw.text(
            (label_x + 1, label_y),
            label_text,
            fill=TEXT_COLOR,
            font=font
        )

    return annotated_image

if __name__ == "__main__":
    server_url = "http://pool0-01436:8000/parse"
    image_path = "../../examples/screenshots/hello_world.png"

    result = test_parsing(server_url, image_path)
    image = Image.open(image_path)
    annotated_image = draw_bboxes(image, result["boxes"])
    annotated_image.save(f"./annotated_images/{Path(image_path).stem}_annotated.png")

    ipdb.set_trace()
    pass



    # # offline test
    # from rfdetr.detr import RFDETRMedium
    # from PIL import Image
    # import io
    #
    # model = RFDETRMedium(
    #     pretrain_weights="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/UI-DETR-1/model.pth",
    #     resolution=1600
    # )
    #
    # # process image
    # image_path = "../../examples/screenshots/click.png"
    # with open(image_path, "rb") as f:
    #     image_bytes = f.read()
    # images = [Image.open(io.BytesIO(image_bytes)).convert("RGB")] * 2
    #
    # # run inference
    # detections_list = model.predict(images, threshold=0.3)
    #
    # ipdb.set_trace()
    # pass
