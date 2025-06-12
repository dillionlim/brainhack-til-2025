"""Manages the OCR model."""

import cv2
import numpy as np
import time
import torch

from doctr.models import crnn_mobilenet_v3_large, fast_base, ocr_predictor
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
    
    def __init__(self, warmup: bool = True):
        # This is where you can initialize your model and any static configurations.
                
        reco_model = crnn_mobilenet_v3_large(pretrained=False, pretrained_backbone=False)
        reco_params = torch.load('crnn_mobilenet_v3_large_3mv2.pt', map_location="cpu")
        reco_model.load_state_dict(reco_params)
        
        det_model = fast_base(pretrained=False, pretrained_backbone=False)
        det_params = torch.load('fast_base_10e.pt', map_location="cpu")
        det_model.load_state_dict(det_params)
        self.model = ocr_predictor(pretrained=True, det_arch = det_model, reco_arch = reco_model, det_bs= 4, reco_bs = 256)
        
        if torch.cuda.is_available():
            self.model.cuda().half()
            print("USING GPU")
        else:
            print("USING CPU")
                        
        if not warmup:
            return
        
        # WARMUP
        with open('testing.jpg', 'rb') as f:
            img_bytes = f.read()
        
        start = time.time()
        print(self.ocr([img_bytes]))
        end = time.time()
        
        print(end - start)

        return
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Mask to remove ghost text
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
        img[img>thresh] = 255
    
        # Make it fLavourless 
        img = cv2.medianBlur(img, 3)

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
                
        texts = [
            ' '.join([
                lines[1].strip().strip('-') 
                for lines in doc_lines
            ]) 
            for doc_lines in docs_lines
        ]

        return texts
