
```bash
trtexec --onnx=crnn_vgg16_bn.onnx \
    --minShapes=input:1x3x32x128 \
    --optShapes=input:256x3x32x128 \
    --maxShapes=input:256x3x32x128 \
    --saveEngine=crnn_vgg16_bn.engine \
    --fp16 \
    --workspace=1024
```

```bash
trtexec --onnx=fast_base.onnx \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:2x3x1024x1024 \
    --maxShapes=input:2x3x1024x1024 \
    --saveEngine=fast_base.engine \
    --fp16 \
    --workspace=1024
```
trtexec --onnx=fast_base.onnx --saveEngine=fast_base.trt


