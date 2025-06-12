"""Manages the CV model."""


from typing import Any
from ultralytics import YOLO, RTDETR
import torch
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

class CVManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(self.device)
        self.detector = YOLO("v8l_averaged_model.pt", task="detect")
        self.imgsz = (1088, 1920)
        
        try:
            dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.detector(dummy_img, imgsz=self.imgsz, half=True, device=self.device, verbose=False)
            print("Models warmed up")
        except Exception as e:
            print(f"Error during model warmup: {e}")
            
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
        detections = self.detector(image, conf=0.5, iou=0.35, agnostic_nms=False, imgsz=self.imgsz, device=self.device, verbose=False, half=True)[0].boxes
        predictions = []
        for xyxy, class_id in zip(detections.xyxy, detections.cls):
            xywh = self.xyxy_to_xywh(xyxy)
            class_id = int(class_id)
            
            predictions.append(
                {
                    "bbox": xywh,
                    "category_id": class_id
                }
            )
        return predictions
    
    @staticmethod
    def xyxy_to_xywh(xyxy) -> list:
        x1, y1, x2, y2 = xyxy
        return [float(x1), float(y1), float(x2-x1), float(y2-y1)]
