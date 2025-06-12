import os

os.environ["RECOGNITION_BATCH_SIZE"] = "1"
os.environ["DETECTOR_BATCH_SIZE"] = "4"

import io
import cv2
import numpy as np

from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

path = os.path.join(os.path.expanduser('~'), 'advanced/ocr/sample_1000.jpg') 
with open(path, 'rb') as file:
    img_bytes = file.read()
    
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

detection_predictor = DetectionPredictor()
recognition_predictor = RecognitionPredictor()

detections = detection_predictor([img])
bboxes = [[i.bbox for i in img_layout.bboxes] for img_layout in detections]
    
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
for i, bbox in enumerate(bboxes[0]):
    x1, y1, x2, y2 = bbox
    print(bbox)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(
        img, str(i+1), (int(x1), int(y1) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, (255, 0, 0), 1, cv2.LINE_AA
    )
    
cv2.imwrite("debug_input.jpg", img)
