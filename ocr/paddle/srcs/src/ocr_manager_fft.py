"""Manages the OCR model."""

import cv2
import numpy as np
import os

from functools import cmp_to_key
from paddleocr import PaddleOCR
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
        
        self.model = PaddleOCR(
            det_model_dir='../en_PP-OCRv3_det/', # change when building docker to './en_PP-OCRv3_det/'
            rec_model_dir='../en_PP-OCRv4_rec/', # change when building docker to './en_PP-OCRv3_det/'               
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
        
        self.pixel_threshold = 200

        pass
    
    def fft_bandpass_filter(self, img, low_cut=0.5, high_cut=300):
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape

        # FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # Create bandpass mask
        crow, ccol = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)

        def soft_bandpass_mask(h, w, low_cut, high_cut):
            crow, ccol = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)
            low_pass = np.exp(-((distance / high_cut)**2))
            high_pass = 1 - np.exp(-((distance / low_cut)**2))
            return low_pass * high_pass

        mask = soft_bandpass_mask(h, w, low_cut, high_cut)

        # Apply mask
        fshift_filtered = fshift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize to uint8
        img_back = np.clip(img_back, 0, 255)
        img_back = img_back / img_back.max() * 255
        img_back = img_back.astype(np.uint8)

        bgr_image =  cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        inverted_image = cv2.bitwise_not(bgr_image)
        inverted_grayscale_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

        return inverted_image.astype(np.uint8)
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # img[img > self.pixel_threshold] = 255 # lightening mask to remove ghost text
        
        denoised = cv2.medianBlur(img, 3)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast, 1.2, blurred, -0.2, 0)  # Moderate sharpening

        # Edge highlighting (using Laplacian)
        edges = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
        edge_enhanced = cv2.addWeighted(sharpened, 1.0, edges, 0.2, 0)  # 0.2 = 20% edge overlay
        
        img = cv2.resize(edge_enhanced, (0, 0), fx=0.5, fy=0.5)
        
        # img = self.fft_bandpass_filter(img) # fft bandpass filter to remove ghost text
        
        return img

    def ocr(self, image_bytes: bytes) -> str:
        """Performs OCR on an image of a document.

        Args:
            image_bytes: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """
        
        img = self.preprocess(image_bytes)
        
        result = self.model.ocr(img, cls = False)
        
        if result[0] is None: 
            # cv2.imwrite("output.jpg", img)
            return ''
        
        lines_data = [
            (
                [poly[0][0], poly[0][1], poly[2][0], poly[2][1]], 
                text,
                conf
            ) for poly, (text, conf) in result[0]
        ]
            
        lines_data = sorted(lines_data, key=compare_lines)
                
        text = ' '.join([line_data[1] for line_data in lines_data])

#         for i, (bbox, _, conf) in enumerate(lines_data):
#             bbox = [int(n) for n in bbox]
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 img, 
#                 f"{str(i+1)} {conf}",
#                 (x1, y1 - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.5, (255, 0, 0), 1, cv2.LINE_AA
#             )

#         cv2.imwrite("output.jpg", img)
        
        return text
