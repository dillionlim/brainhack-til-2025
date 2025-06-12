"""Manages the OCR model."""

import cv2
import io
import numpy as np
import os

os.environ["RECOGNITION_BATCH_SIZE"] = "8"
os.environ["DETECTION_BATCH_SIZE"] = "4"
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
os.environ["TORCH_DEVICE"] = "cuda"

from functools import cmp_to_key
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from typing import List


def arrange_lines(bbox1: List[int], bbox2: List[int]) -> bool:
    if bbox1[2] < bbox2[0]:
        return -1
        
    if bbox2[2] < bbox1[0]:
        return 1
    
    return bbox1[1] - bbox2[1] 

    
compare_lines = cmp_to_key(arrange_lines)


class OCRManager:
    
    def __init__(self):
        
        self.detector = DetectionPredictor()
        self.recogniser = RecognitionPredictor()
        
        self.pixel_threshold = 200
        
    def preprocess(self, image_bytes: bytes) -> Image:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

        # light_mask = img_gray > pixel_threshold
        # img_gray[light_mask] = 255

        return Image.fromarray(img_gray)

    def ocr(self, images_bytes: List[bytes]) -> str:
        """Performs OCR on an image of a document.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """
        
        imgs = [self.preprocess(image_bytes) for image_bytes in images_bytes]
        detections = self.detector(imgs)
        imgs_lines = [[i.bbox for i in img_text.bboxes] for img_text in detections]
        imgs_lines = [sorted(img_lines, key=compare_lines) for img_lines in imgs_lines]
        
        outputs = self.recogniser(imgs, bboxes = imgs_lines)
        
        predictions = [
            ' '.join([line.text for line in output.text_lines])
            for output in outputs
        ]
        
        return predictions
