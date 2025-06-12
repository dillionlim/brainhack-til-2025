import io
import os
import numpy as np
import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from paddleocr import PaddleOCR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = PaddleOCR(
    det_model_dir='../src/en_PP-OCRv3_det/',
    lang='en',
    rec=False,
    use_gpu=True,
    use_angle_cls=False
) 

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', use_fast = True)
recogniser = VisionEncoderDecoderModel.from_pretrained('../output/trocr/best').to(device)
recogniser.eval()

print("LOADED")

path = os.path.join(os.path.expanduser('~'), 'advanced/ocr/sample_1000.jpg') 
with open(path, 'rb') as file:
    img_bytes = file.read()
    
result = detector.ocr(img_bytes, cls = False)

img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

cropped_imgs = []
for box, _ in result[0]:
    pts = np.array(box).astype(int)
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])
    
    cropped = img.crop((x_min, y_min, x_max, y_max))
    cropped_imgs.append(cropped)
    
inputs = processor(images=cropped_imgs, return_tensors="pt", padding=True)
pixel_values = inputs.pixel_values.to(device)

with torch.no_grad():
    generated_ids = recogniser.generate(pixel_values)

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
transcript = ' '.join([text if text else '' for text in generated_texts])
    
print(transcript)
    