"""Manages the OCR model."""

import cv2
import numpy as np
import os
import re

from functools import cmp_to_key
from paddleocr import PaddleOCR
from symspellpy import SymSpell, Verbosity
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
    DEFAULT_COUNT = 500000000

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        
        self.model = PaddleOCR(
            det_model_dir='../en_PP-OCRv3_det/', # change when building docker to './en_PP-OCRv3_det/'
            rec_model_dir='../en_PP-OCRv4_rec/', # change when building docker to './en_PP-OCRv3_det/'
            # rec_char_dict_path='./en_dict.txt',
            # rec_image_shape="3, 48, 320",
            cls_model_dir=None,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            rec_batch_num=3,
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
        
        # Load Symspell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        dictionary_path = "symspell_dicts/frequency_dictionary_en_82_765.txt"
        bigram_path = "symspell_dicts/frequency_bigramdictionary_en_243_342.txt"
    
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        # Load commonly used words in dataset
        with open('symspell_dicts/unigram_dict.txt', 'r') as f:
            data = f.read()

        data = data.split('\n')

        for x in data:
            word = x.split('\t')[0]
            suggestion = self.sym_spell.lookup(word, Verbosity.CLOSEST)

            if not suggestion:
                self.sym_spell.create_dictionary_entry(word, self.DEFAULT_COUNT)
                continue

            if suggestion[0].count < self.DEFAULT_COUNT:
                self.sym_spell.create_dictionary_entry(word, self.DEFAULT_COUNT)

        # Prepare proper nouns
        self.proper_nouns = [
            "CYPHER",
            "Project Quantum Sentinel",
            "BH-2000",
            "Operation Iron Claw"
        ]
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        img[img > self.pixel_threshold] = 255 # lightening mask to remove ghost text
        
        denoised = cv2.medianBlur(img, 3)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast, 1.2, blurred, -0.2, 0)  # Moderate sharpening

        # Edge highlighting (using Laplacian)
        edges = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
        img = cv2.addWeighted(sharpened, 1.0, edges, 0.2, 0)  # 0.2 = 20% edge overlay
        
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                
        return img

    def ocr(self, image_bytes: bytes) -> str:
        """Performs OCR on an image of a document.

        Args:
            image_bytes: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """
        
        # Preprocess image
        img = self.preprocess(image_bytes)
        
        # Conduct OCR
        result = self.model.ocr(img, cls = False)
        
        if result[0] is None: 
            # cv2.imwrite("output.jpg", img)
            return ''
        
        # Layout arranging
        lines_data = [
            (
                [poly[0][0], poly[0][1], poly[2][0], poly[2][1]], 
                text,
                conf
            ) for poly, (text, conf) in result[0]
        ]
        
        lines_data = sorted(lines_data, key=compare_lines)
                
        text = ''.join([line_data[1] for line_data in lines_data])

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

        # Regex to remove unnecessary periods
        text = re.sub(r'\. (?=[a-z])', ', ', text)
        text = re.sub(r'(?<=[a-z])\.(?=[a-z])', ' ', text)
        # text = re.sub(r'([a-z])\.([a-z])', r'\1\2', text)
        
        # Preserve punctuations
        parts = re.split(r'(?<=[.!?,:])\s+', text)
        punctuations = [part[-1] for part in parts]

        # Use symspellpy spell corrector
        suggestions = []
        for part in parts:
            suggestions.append(self.sym_spell.lookup_compound(part.strip(), max_edit_distance=2, transfer_casing = True)[0])
                                                    
        # Reintegrate punctuations
        parts = [x.term for x in suggestions] 
        for i in range(len(parts)):
            parts[i] = parts[i] + punctuations[i]

        text = ' '.join(parts)

        # Properly capitalise key proper nouns
        for p_noun in self.proper_nouns:
            pattern = re.compile(rf'\b{re.escape(p_noun)}\b', re.IGNORECASE)
            text = pattern.sub(p_noun, text)
        
        return text
