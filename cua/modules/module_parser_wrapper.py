from typing import List

from PIL import Image


class ParserWrapper:
    """
    Wrapper class that supports all interaction with remote screen-parser server.
    """
    def __init__(self):
        # todo
        pass

    def parse_screenshot(self, image_bytes: bytes, ):
        # todo call parser server to retrieve bbox for all interactable marks
        # todo the set-of-marks processing should be done on the client side, using prepare_image_with_som

        pass

    def prepare_image_with_som(self, image: Image.Image, bbox_list: List):
        # given bbox_list and image, mark all bboxes on the image
        pass



