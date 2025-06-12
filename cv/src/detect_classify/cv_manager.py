"""Manages the CV model."""


from typing import Any
from ultralytics import YOLO
from swinir import SwinIR
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from io import BytesIO
from PIL import Image
import numpy as np

class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '2', '3', '4', '5', '6', '7', '8', '9']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(self.device)
        self.detector = YOLO("yolov8_detector.pt", task="detect")
        self.detector_IMGSZ = (1088, 1920)
        self.upscaler = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        pretrained_swinir = torch.load("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", weights_only=True) if torch.cuda.is_available() else torch.load("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", weights_only=False)
        self.upscaler.load_state_dict(pretrained_swinir['params_ema'], strict=True)
        self.upscaler.eval()
        self.upscaler = self.upscaler.to(self.device)
        self.classifier_dim = 224
        self.classifier = resnet50()
        num_ftrs_resnet = self.classifier.fc.in_features
        self.classifier.fc = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(num_ftrs_resnet, len(self.class_names))
                                )
        resnet50_state_dict = torch.load("resnet50_finetuned.pth", weights_only=True) if torch.cuda.is_available() else torch.load("resnet50_finetuned.pth", weights_only=False)
        self.classifier.load_state_dict(resnet50_state_dict, strict=True)
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device)
        self.classifier_transforms = transforms.Compose([
            transforms.Resize((self.classifier_dim, self.classifier_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            dummy_det = [np.zeros((self.detector_IMGSZ[0], self.detector_IMGSZ[1], 3), dtype=np.uint8) for i in range(4)]
            self.detector(dummy_det, imgsz=self.detector_IMGSZ, batch=4, verbose=False, device=self.device)
            dummy_cls = Image.fromarray(np.zeros((self.classifier_dim, self.classifier_dim, 3), dtype=np.uint8))
            dummy_cls = self.classifier_transforms(dummy_cls).unsqueeze(0).cuda()
            with torch.no_grad():
                print(self.classifier(dummy_cls))
            dummy_sr = np.zeros((64, 64, 3), dtype=np.uint8)
            self.upscale_image(dummy_sr) # Call the single image upscaler
            print("Models warmed up.")
        except Exception as e:
            print(f"Warning during model warmup: {e}")

    async def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        # Your inference code goes here.
        image_bytes = BytesIO(image)
        image = Image.open(image_bytes)
        image.load()
        detections = self.detector(image, verbose=False)[0].boxes
        predictions = []
        for xyxy, superclass_id in zip(detections.xyxy, detections.cls):
            superclass_id = int(superclass_id)
            xyxy = xyxy.cpu().numpy()
            xywh = self.xyxy_to_xywh(xyxy)
            cropped_img = image.crop(xyxy)
            if cropped_img.width < 128 and cropped_img.height < 128: # guard against CUDA OOM
                cropped_img = Image.fromarray(self.upscale_image(np.asarray(cropped_img)))
            class_id = self.get_class_from_superclass(superclass_id, cropped_img)
            predictions.append(
                {
                    "bbox": xywh,
                    "category_id": int(class_id)
                }
            )
        # print(predictions)
        return predictions
    
    @staticmethod
    def xyxy_to_xywh(xyxy) -> list:
        x1, y1, x2, y2 = xyxy
        return [float(x1), float(y1), float(x2-x1), float(y2-y1)]
    
    @torch.no_grad()
    @torch.inference_mode()
    def get_class_from_superclass(self, superclass_id: int, cropped_img: Image) -> int:
        cropped_img = self.classifier_transforms(cropped_img).unsqueeze(0).cuda()
        _, pred = torch.max(self.classifier(cropped_img), 1)
        return int(self.class_names[pred.item()])
    
    @torch.no_grad()
    @torch.inference_mode()
    def upscale_image(self, img_lq):
        img_lq = img_lq.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                                              (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
        window_size = 8
        scale = 4
        
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = self.upscaler(img_lq)
        output = output[..., :h_old * scale, :w_old * scale]
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)
        return output
