# OCR Rundown

The OCR task involves reading text in a scanned document. The scanned documents have different layouts and its text has different fonts. Most of the scanned documents also have some or all of the following three augmentations:

1. Salt and pepper noise
2. Blurred text
3. Mirrored 'ghost text' pasted on the original text

### Preprocessing

To handle the augmentations and the layout, we preprocessed all images using **median blur** (to remove salt and pepper noise) and **OTSU thresholding** (to remove as much 'ghost text' as possible without degrading the original text too much). We tried **CLAHE** (Contrast Limited Adaptive Histogram Equalization) as well for enhancing constrast and although the image looked slightly clear the score did not improve with it. 

### Postprocessing

Most OCR libraries return the lines of text sorted top-down. However, as documents have 2 columns, this will return the text in the wrong order. To handle this, we used a simple comparison to re-order the lines. 

```python 
def arrange_lines(line1: LinesData, line2: LinesData) -> bool:
    bbox1 = line1[0]
    bbox2 = line2[0]
    
    if bbox1[2] < bbox2[0]:
        return -1
        
    if bbox2[2] < bbox1[0]:
        return 1
    
    return bbox1[1] - bbox2[1] 
```

This will not work for more complex layouts (e.g. those found in physical newspapers).

### Models Tried

We tried a variety of baseline models. Some such as pytesseract, surya-ocr and trocr were able to achieve decent accuracy but were extremely slow (surya-ocr timed out). PaddleOCR and docTR were able to achieve high accuracy and was much faster. As docTR had higher accuracy and slightly better speed, we used it for the semi-finals.

| Detector Model | Dataset | Training Args | Recogniser Model | Dataset | Training Args | Score | Speed |
|:--------:|:------------:|:----------------:|:----------------:|:------------:|:----------------:|:-----:|:-----:|
| `fast_base`    | Default (preprocessed) | 10 epochs, LR 0.0001, batch size 2 | `crnn_vgg16_bn` | 3 epochs default + 3 epochs Mixedv2 | 3 epochs, LR 0.00002, batch size 256, freeze backbone + 5 epochs LR 0.00001, batch size 128, unfreeze backbone | 0.983 | 0.779 |
| `fast_base`    | Default (preprocessed) | 10 epochs, LR 0.0001, batch size 2 | `crnn_mobilenet_v3_large` | Mixedv2 | 3 epochs LR 0.00001, batch size 128, unfreeze backbone | 0.981 | 0.844 |

For the full list of models submitted, training args and evaluation scores, refer to the model tracking google sheet.

We used `crnn_mobilenet_v3_large` for the semi-finals as it is significantly faster for only a slight dip in accuracy. 

#### Training

For text detector, we fine-tuned it using the full default dataset. However, as we intend to do preprocessing for inference, we also preprocessed the default dataset for the model to train on.

For text recogniser, we initially tried fine-tuning it with the full dataset. However, we realised that the number of unique ground truths for the provided dataset is quite limited (only 5 unique ground truths). To prevent overfitting, we generated our own dataset with text from ChatGPT. 

For docTR, text recognition has to be fine-tuned on images of **words** while with PaddleOCR, text recognition can be fine-tuned on images of **lines/phrases**

The generation and preperation of datasets can be found in the following files:
- [docTR](doctr/train/data_prep/dataset_prep.ipynb) 
- [paddleocr](paddle/train/dataset_prep.ipynb)
