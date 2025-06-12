**NOTE: ONNX ENDED UP SLOWLY THAN JUST USING GPU**, but if you want to install, you have to install `paddlepaddle-gpu==3.0.0` from this wheel (https://paddle-whl.bj.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.0.0-cp312-cp312-linux_x86_64.whl) and then install `paddle2onnx`. 

You also have to git clone the [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)

```bash 
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

### RECOGNITION

#### Training

```bash
uv run python PaddleOCR/tools/train.py -c train_config.yml -o Global.pretrained_model=pretrained/en_PP-OCRv4_rec_train/best_accuracy
```

#### Exporting as inference

```bash
uv run python3 PaddleOCR/tools/export_model.py \
  -c train_config.yml \
  -o Global.pretrained_model=./output/rec_ppocr_v4/best_accuracy \
  Global.save_inference_dir=./srcs/en_PP-OCRv4_rec/
```

#### Exporting as onnx

```bash
uv run paddle2onnx --model_dir ./srcs/en_PP-OCRv4_rec \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ../srcs/onnx/en_PP-OCRv4_rec.onnx \
    --opset_version 11 \
    --enable_onnx_checker True
```


### DETECTION

#### Training

```bash
uv run python3 PaddleOCR/tools/train.py -c train_config_det.yml -o Global.pretrained_model=./pretrained/en_PP-OCRv3_det_train/MobileNetV3_large_x0_5_pretrained
```  

#### Exporting as inference

```bash
uv run python3 PaddleOCR/tools/export_model.py \
  -c PaddleOCR/configs/det/det_mv3_db.yml \
  -o Global.pretrained_model=./output/det_ppocr_v3/best_accuracy \
  Global.save_inference_dir=../srcs/en_PP-OCRv3_det/
```
  
#### Exporting as onnx

```bash
uv run paddle2onnx --model_dir ./srcs/en_PP-OCRv3_det \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ../srcs/onnx/en_PP-OCRv3_det.onnx \
    --opset_version 11 \
    --enable_onnx_checker True
```
