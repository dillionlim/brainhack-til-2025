from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from typing import Union, List, Tuple
import PIL
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
import torchvision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class BaseModel:
    def __init__(self, imgsz: Union[Tuple[int], List[int], int] = (1088, 1920), half: bool = True,
                 conf: float = 0.3, iou: float = 0.6, device: Union[torch.device, int, str] = 0):
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else: self.imgsz = imgsz
        self.half = half
        self.conf = conf
        self.iou = iou
        self.device = device
    
        
class YOLOModel(BaseModel):
    '''Class for YOLO models which process the entire image'''
    def __init__(self, model: str, imgsz: Union[Tuple[int], List[int], int] = (1088, 1920), half: bool = True,
                 conf: float = 0.3, iou: float = 0.6, device: Union[torch.device, int, str] = 0):
        super().__init__(imgsz, half, conf, iou, device)
        
        self.model = YOLO(model, task="detect")
        
        try: # Warm up model
            dummy_img = np.zeros((*self.imgsz, 3), dtype=np.uint8)
            self.model(dummy_img, imgsz=self.imgsz, half=self.half, device=self.device, verbose=False)
            print("Model warmed up")
        except Exception as e:
            print(f"Error during model warmup: {e}")
            
    def __call__(self, image: Union[PIL.Image.Image, np.ndarray, str], verbose: bool = False) -> Boxes:
        results = self.model.predict(image, imgsz=self.imgsz, half=self.half, conf=self.conf, iou=self.iou, device=self.device, verbose=verbose)
        # return results[0].boxes
        return results[0].boxes.xyxyn, results[0].boxes.conf, results[0].boxes.cls

    
class SAHIModel(BaseModel):
    def __init__(self, model: str, category_mapping: dict[str, str], imgsz: Union[Tuple[int], List[int], int] = (800, 800), 
                 conf: float = 0.4, iou: float = 0.6, postprocess_type="GREEDYNMM", postprocess_match_threshold=0.5, device: Union[torch.device, int, str] = 0):
        assert model.endswith(".onnx"), "Export model to onnx"
        super().__init__(imgsz, conf, iou, device)
        self.postprocess_type = postprocess_type
        self.postprocess_match_threshold = postprocess_match_threshold
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8onnx", # seems to work for all onnx
            model_path=model,
            confidence_threshold=self.conf,
            device=self.device,
            image_size=self.imgsz,
            category_mapping=category_mapping,
            iou_threshold=self.iou
        )
    def __call__(self, image: Union[PIL.Image.Image, np.ndarray, str], overlap_height_ratio=0.64, overlap_width_ratio=0.3, verbose=0):
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=self.imgsz[0],
            slice_width=self.imgsz[1],
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric="IOS",
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=True,
            verbose=verbose
        )
        predictions = []
        xyxyn = []
        conf = []
        cls_ = []
        for detection in result.object_prediction_list:
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            class_id = detection.category.id
            det_conf = detection.score.value
            xyxyn.append([x1 / result.image_width, y1 / result.image_height, x2 / result.image_width, y2 / result.image_height])
            conf.append(det_conf)
            cls_.append(class_id)
        return xyxyn, conf, cls_
        
class DetectorEnsemble:
    def __init__(self, models: List[YOLOModel], model_weights: List[int], 
                 iou_thr: float = 0.3, skip_box_thr: float = 0.3):
        assert len(models) == len(model_weights), "Number of models must equal number of model weights"
        self.models = models
        self.model_weights = model_weights
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
    
    def __call__(self, image: Union[PIL.Image.Image, np.ndarray, str], final_conf: float = 0.3, verbose: bool = False):
        '''Use weighted box fusion to ensemble model detections, then perform class agnostic NMS to remove double predictions
        '''
        if isinstance(image, PIL.Image.Image):
            IMG_WIDTH, IMG_HEIGHT = img.size
        else: # np array of shape (IMG_HEIGHT, IMG_WIDTH, 3)
            IMG_HEIGHT, IMG_WIDTH = image.shape[:2]
            
        results = [model(image) for model in self.models]
        boxes_list, scores_list, labels_list = [], [], []
        for xyxyn, conf, cls_ in results:
            boxes_list.append(xyxyn)
            scores_list.append(conf)
            labels_list.append(cls_)
        if all([len(scores)==0 if isinstance(scores, list) else scores.shape == torch.Size([0]) for scores in scores_list]):
            # No detections
            return []
        
        # WBF
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=self.model_weights, iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr, conf_type='avg')
        
        # Class agnostic NMS
        keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=self.iou_thr).cpu().numpy()
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if not scores.shape:
            scores = np.expand_dims(scores, 0)
            labels = np.expand_dims(labels, 0)
        
        final_predictions = []
        for box, score, label in zip(boxes, scores, labels):
            if score < final_conf: continue
            x1, y1, x2, y2 = box[0]*IMG_WIDTH, box[1]*IMG_HEIGHT, box[2]*IMG_WIDTH, box[3]*IMG_HEIGHT
            w, h = x2-x1, y2-y1
            final_predictions.append(
                {
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "category_id": int(label)
                }
            )
        return final_predictions