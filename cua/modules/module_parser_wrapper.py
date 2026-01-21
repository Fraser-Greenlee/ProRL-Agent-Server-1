class ParserWrapper:
    """
    Wrapper class that supports all interaction with remote screen-parser server.
    """
    def __init__(self):
        # todo
        pass

    def parse_screenshot(self):
        # todo call parser server to retrieve bbox for all interactable marks
        # todo the set-of-marks processing should be done on the client side, using prepare_image_with_som
        pass

    def prepare_image_with_som(self, bbox_list, image):
        # given bbox_list and image, mark all bboxes on the image
        pass



