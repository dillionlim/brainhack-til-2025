import os 
import numpy as np
import cv2


def preprocess(image_bytes: bytes) -> np.ndarray:

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Mask to remove ghost text
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    thresh = min(210, thresh)
    img[img>thresh] = 255
    
    # Median blur to remove salt and pepper noise
    img = cv2.medianBlur(img, 3)    

#     # Contrast Limited Adaptive Histogram Equalization (to enhance contrast)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     contrast = clahe.apply(img)
#     blurred = cv2.GaussianBlur(contrast, (3, 3), 1.0)
#     img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)  # Moderate sharpening

#     # Laplacian to highlight edges
#     edges = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
#     img = cv2.addWeighted(img, 1.0, edges, 0.2, 0)  # 0.2 = 20% edge overlay

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img


for x in os.listdir('problems'):
    if not x.startswith('sample'):
        continue
    
    with open(os.path.join('problems', x), 'rb') as f:
        img_b = f.read()
        nparr = np.frombuffer(img_b, np.uint8)
    nparr = preprocess(nparr)
    cv2.imwrite(os.path.join('problems', f"p_{x}"), nparr)
