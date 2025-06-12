import torch
from doctr.models import crnn_vgg16_bn, fast_base
from doctr.models.utils import export_model_to_onnx

batch_size = 1
input_shape_det = (3, 1024, 1024)
input_shape_rec = (3, 32, 128)

det_model = fast_base(pretrained=True, exportable=True)
rec_model = crnn_vgg16_bn(pretrained=True, exportable=True)

dummy_input_det = torch.rand((batch_size, *input_shape_det), dtype=torch.float32)
dummy_input_rec = torch.rand((batch_size, *input_shape_rec), dtype=torch.float32)

det_model_path = export_model_to_onnx(
    det_model,
    model_name="../src/fast_base",
    dummy_input=dummy_input_det
)
rec_model_path = export_model_to_onnx(
    rec_model,
    model_name="../src/crnn_vgg16_bn",
    dummy_input=dummy_input_rec
)
