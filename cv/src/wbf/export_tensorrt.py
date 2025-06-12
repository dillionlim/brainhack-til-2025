from ultralytics import YOLO
import glob

def export_tensorrt(yolo_pt_path):
    model = YOLO(yolo_pt_path)
    model.export(format="engine", imgsz=(1088, 1920), device=0, half=True, nms=False)
    
model_paths = glob.glob("models/*.pt")

for path in model_paths:
    export_tensorrt(path)