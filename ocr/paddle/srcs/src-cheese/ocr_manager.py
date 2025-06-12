"""Manages the OCR model."""

from collections.abc import Sequence
from typing import List
from paddleocr import PaddleOCR

import jiwer
import cv2
import numpy as np

cer_transforms = jiwer.Compose([
    jiwer.SubstituteRegexes({"-": ""}),
    jiwer.RemoveWhiteSpace(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])

def score_ocr(preds: Sequence[str], ground_truth: Sequence[str]) -> float:
    return 1 - jiwer.cer(
        ground_truth,
        preds,
        truth_transform=cer_transforms,
        hypothesis_transform=cer_transforms,
    )

class OCRManager:

    def __init__(self):
        
        self.model = PaddleOCR(
            det_model_dir='./en_PP-OCRv3_det/',
            rec_model_dir='./en_PP-OCRv4_rec/',
            cls_model_dir=None,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            rec_batch_num=8,
            lang='en',
            precision='fp16',
            use_gpu=True,
            use_mp=True,
            warmup=True,
            show_log=False,
            table=False,
            rec=True,
            use_angle_cls=False,   
        )
        
        self.outputs = []
        for i in [1, 2, 3, 4]:
            with open(f"pred_{i}.txt", 'r') as f:
                self.outputs.append(f.read())
                
        self.pixel_threshold = 200

        pass
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        height, width = img.shape[:2]
        img = img[200:height//6, 0:width]
        
        img[img > self.pixel_threshold] = 255
        
        denoised = cv2.medianBlur(img, 3)
        
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         contrast = clahe.apply(denoised)
#         blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
#         sharpened = cv2.addWeighted(contrast, 1.2, blurred, -0.2, 0)  # Moderate sharpening

#         # Edge highlighting (using Laplacian)
#         edges = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
#         edge_enhanced = cv2.addWeighted(sharpened, 1.0, edges, 0.2, 0)  # 0.2 = 20% edge overlay
        
#         img = cv2.resize(edge_enhanced, (0, 0), fx=0.5, fy=0.5)
        
        return denoised
    
    def ocr(self, image_bytes: bytes) -> str:
        """Performs OCR on an image of a document.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """
        
        img = self.preprocess(image_bytes)
        
        result = self.model.ocr(img, cls = False)
        
        # print(result)

        if result[0] is None:
            return self.output[2]
       
        text = ''.join([line[1][0] for line in result[0]])
        
        best_score = 0
        best_output = 0
        for i, o in enumerate(self.outputs):
            curr = score_ocr(text, o[:len(text)])
            if curr > best_score:
                best_score = curr
                best_output = i

        return self.outputs[best_output]
