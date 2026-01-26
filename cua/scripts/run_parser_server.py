import ipdb
import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File
from typing import List
import io
import torch
from PIL import Image
import numpy as np
from ray.util import rpdb

# --- Import your specific model ---
# Ensure rfdetr is installed in your environment
from rfdetr.detr import RFDETRMedium
from supervision import Detections

app = FastAPI()


@serve.deployment(
    num_replicas=8,  # <--- SCALING: Launches 8 independent workers
    ray_actor_options={"num_gpus": 1},  # <--- HARDWARE: Each worker gets 1 exclusive GPU
    max_ongoing_requests=256  # Allow queue to build up to enable batching
)
@serve.ingress(app)
class RFDETRDeployment:
    def __init__(self):
        """
        This method runs ONCE per worker process when it starts.
        Because we requested "num_gpus": 1, Ray sets CUDA_VISIBLE_DEVICES
        such that each worker sees exactly one unique GPU as 'cuda:0'.
        """
        print(f"Initializing model on worker. Visible devices: {torch.cuda.device_count()}")

        # Initialize the model instance for this specific GPU worker
        self.model = RFDETRMedium(
            pretrain_weights="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/UI-DETR-1/model.pth",
            resolution=1600
        )

        # Warmup (Optional but recommended to allocate VRAM immediately)
        # self.model.predict([Image.new('RGB', (100, 100))], threshold=0.5)
        print("RF-DETR Model loaded and ready.")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def handle_batch(self, image_bytes_list: List[bytes]):
        """
        This function receives a LIST of raw image bytes from multiple concurrent requests.
        Ray Serve automatically groups them for us.
        """

        # 1. Convert raw bytes to PIL Images (CPU Bound - runs in parallel with other workers)
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

        # 2. Run Batch Inference (GPU Bound)
        detections_list = self.model.predict(images, threshold=0.15)

        # if the request was single image, detections_list is not a list but a single Detections object
        if isinstance(detections_list, Detections):
            detections_list = [detections_list]

        # 3. Serialize Results
        # We must return a list of results strictly matching the order of input images.
        results = []
        for detections in detections_list:
            # detections object contains: xyxy, confidence, class_id
            # We need to convert numpy/tensor arrays to standard Python lists for JSON

            # Extract boxes (xyxy)
            boxes = detections.xyxy.tolist()

            # Extract scores (confidence)
            scores = detections.confidence.tolist()

            results.append({
                "boxes": boxes,
                "scores": scores,
                "count": len(boxes)
            })

        return results

    @app.post("/parse")
    async def parse(self, file: UploadFile = File(...)):
        """
        The public endpoint.
        The client calls this function individually.
        Ray intercepts the call, buffers the 'file', sends it to 'handle_batch',
        waits for the result, and returns it to this specific client.
        """
        image_bytes = await file.read()

        # Await the batch handler
        # This looks like a single call, but it's actually entering the batch queue
        result = await self.handle_batch(image_bytes)

        return result


if __name__ == "__main__":
    # 1. Start the Serve System with 0.0.0.0
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    # 2. Deploy the application
    # Bind the deployment to the application
    parser_app = RFDETRDeployment.bind()
    serve.run(parser_app)

    # 3. Keep the script running (serve.run is non-blocking)
    print("Server running on http://0.0.0.0:8000")
    import time

    while True:
        time.sleep(10)


# USAGE
# use cua_vllm.sqsh
# ray stop --force && python parser_server.py

