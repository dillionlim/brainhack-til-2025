"""Manages the CV model."""


from typing import Any
import torch
import numpy as np
import cv2
from ensemble import YOLOModel, DetectorEnsemble, SAHIModel

class CVManager:
    def __init__(self):
        self.device = 0
        self.imgsz = (1088, 1920)
        v8m_finetunev1 = YOLOModel("models/yolov8m_1920_finetune.pt", iou=0.6, device=self.device)
        v8m_averaged_v2v3 = YOLOModel("models/v8m_averaged_model.pt", iou=0.6, device=self.device)
        v8l_averaged = YOLOModel("models/v8l_averaged_model.pt", iou=0.6, device=self.device)
        models = [v8m_finetunev1, v8m_averaged_v2v3, v8l_averaged]
        self.detector = DetectorEnsemble(models, [5, 8, 8], iou_thr=0.5, skip_box_thr=0.10)
        
    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        # Your inference code goes here.
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        return self.detector(image, final_conf=0.25)
