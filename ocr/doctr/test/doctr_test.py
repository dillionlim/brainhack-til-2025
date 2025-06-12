import time
import torch

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(
    pretrained=True
).cuda()

start_doc = time.time()
doc = DocumentFile.from_images(["../src/testing.jpg"]*4)
start = time.time()

# with torch.amp.autocast('cuda'):
result = model(doc)

half = time.time()

output = result.export()
for page in output['pages']:
    print("== Page ==")
    for block in page['blocks']:
        for line in block['lines']:
            # Line text
            text = " ".join(word['value'] for word in line['words'])

            # Bounding box: [x_min, y_min, x_max, y_max] (relative coordinates in range [0, 1])
            box = line['geometry']

            # print(line)
            print(f"Line text: {text}")
            print(f"Bounding box: {box}")

end = time.time()
print(start-start_doc)
print(half-start)
print(end-half)
