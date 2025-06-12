"""Manages the OCR model."""

import cv2
import numpy as np
import os
import re
import torch

from onnxtr.models import ocr_predictor, crnn_vgg16_bn, fast_base
from functools import cmp_to_key
from typing import List, Tuple


LinesData = Tuple[List[int], str, float]

def arrange_lines(line1: LinesData, line2: LinesData) -> bool:
    bbox1 = line1[0]
    bbox2 = line2[0]
    
    if bbox1[2] < bbox2[0]:
        return -1
        
    if bbox2[2] < bbox1[0]:
        return 1
    
    return bbox1[1] - bbox2[1] 


compare_lines = cmp_to_key(arrange_lines)


class OCRManager:

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
                
        det_model = fast_base("fast_base.onnx")
        rec_model = crnn_vgg16_bn("crnn_vgg16_bn.onnx")
        
        self.model = ocr_predictor(det_arch = det_model, reco_arch = rec_model)
        self.pixel_threshold = 200
                
        pass
     
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Mask to remove ghost text
        img[img > self.pixel_threshold] = 255 
        
        # Median blur to remove salt and pepper noise
        denoised = cv2.medianBlur(img, 3)
        
        # CLAHE to sharpening image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast, 1.2, blurred, -0.2, 0)  # Moderate sharpening

        # Laplacian to highlight edges
        edges = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
        img = cv2.addWeighted(sharpened, 1.0, edges, 0.2, 0)  # 0.2 = 20% edge overlay
        
        # Resize to increase inferencing speed
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        
        # Convert back to RGB (required by DocTR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
        return img

    def ocr(self, image_bytes_ls: List[bytes]) -> str:
        """Performs OCR on an image of a document.

        Args:
            image_bytes: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """
        
        imgs = [self.preprocess(image_bytes) for image_bytes in image_bytes_ls]
        
        result = self.model(imgs)

        output = result.export()
        
        docs_lines = [
            [ 
                (
                    [line['geometry'][0][0], line['geometry'][0][1], line['geometry'][1][0], line['geometry'][1][1]],
                    ' '.join(word['value'] for word in line['words']),
                    line['objectness_score']
            
                ) for block in page["blocks"] for line in block["lines"]
            ] for page in output["pages"]
        ]
        
        docs_lines = [sorted(doc_lines, key=compare_lines) for doc_lines in docs_lines]
                
        text = [' '.join([lines[1] for lines in doc_lines]) for doc_lines in docs_lines]

#         for i, (bbox, _, conf) in enumerate(docs_lines[0]):
#             bbox = [int(n) for n in bbox]
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(imgs[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 imgs[0], 
#                 f"{str(i+1)} {conf}",
#                 (x1, y1 - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.5, (255, 0, 0), 1, cv2.LINE_AA
#             )

#         cv2.imwrite("output.jpg", imgs[0])

        return text
