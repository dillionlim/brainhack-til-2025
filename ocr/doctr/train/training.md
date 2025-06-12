## Training Scripts for docTR

Scripts are provided by [doctr's repo](https://github.com/mindee/doctr). Run them from the this directory.

### TRAINING RECOGNISER
```bash
uv run python rec/train_pytorch.py crnn_mobilenet_v3_large \
    --resume base_models/crnn_mobilenet_v3_large_pt-f5259ec2.pt \
    --train_path dataset/ocr_words/train \
    --val_path dataset/ocr_words/val \
    --output_dir ./rec/ \
    --lr 0.00001 \
    -b 128 \
    --epochs 3 \
    --vocab french
```

### TRAINING DETECTOR
```bash
uv run python det/train_pytorch.py fast_base \
    --resume base_models/fast_base-688a8b34.pt \
    --output_dir ./det/ \
    --train_path dataset/ocr_detection/train \
    --val_path dataset/ocr_detection/val \
    --lr 0.0001 \
    -b 2 \
    --epochs 10 \
    --early-stop \
    --early-stop-epochs 2
```
