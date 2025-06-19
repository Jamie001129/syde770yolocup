from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO
from PIL import Image
import io
import os
import csv
import time
import json

class CustomHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None
        self.default_conf = 0.6
        self.class_names = {0: "paper_cup", 1: "timmies"}
        self.BASE_DIR = os.getenv("BASE_DIR", "/tmp")
        self.LOG_FILE = os.path.join(self.BASE_DIR, "predictions_log.csv")
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
        # Create log file header if file does not exist or is empty
        if not os.path.exists(self.LOG_FILE) or os.stat(self.LOG_FILE).st_size == 0:
            with open(self.LOG_FILE, mode="w", newline="") as file:
                csv.writer(file).writerow(["filename", "prediction", "latency_ms"])
    
    def initialize(self, ctx):
        # Use the environment variable for the model path if available
        #model_path = "C:/Users/admin/OneDrive - University of Waterloo/SYDE750/projectDVCStorage/Timmies/train7/group1_train_best_weights.torchscript" ####Need to change this to point to the model on onedrive
        model_path = "C:/Users/admin/Documents/UWaterloo/SYDE750/group1_train_best_weights.torchscript"
        self.model = YOLO(model_path, task='detect')
    
    def preprocess(self, data):
        # TorchServe expects the incoming request to contain binary image data under 'data'
        if "data" in data[0] and data[0]["data"]:
            return {
                "image_bytes": data[0]["data"],
                "filename": data[0].get("name", "unknown.jpg")
            }
        else:
            raise ValueError("No image data provided")
    
    def inference(self, data):
        # Directly run prediction on the image data
        return self._predict(data)
    
    def _predict(self, data):
        start_time = time.time()
        image_bytes = data.get("image_bytes")
        if not image_bytes:
            raise ValueError("No image provided for prediction")
        
        # Open the image and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference with the model
        output = self.model(image, conf=self.default_conf)
        predictions = []
        for r in output:
            if not hasattr(r, 'boxes'):
                continue
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            classes = r.boxes.cls.cpu().numpy().tolist()
            for box, score, cls in zip(boxes, scores, classes):
                if score < self.default_conf:
                    continue
                label = self.class_names.get(int(cls), str(cls))
                predictions.append({
                    "label": label,
                    "confidence": round(score, 2),
                    "bbox": [int(coord) for coord in box]
                })
        latency_ms = round((time.time() - start_time) * 1000, 2)
        filename = data.get("filename", "unknown.jpg")
        
        # Log prediction results (if needed)
        with open(self.LOG_FILE, mode="a", newline="") as file:
            csv.writer(file).writerow([filename, json.dumps(predictions), latency_ms])
        
        return {"predictions": predictions}
    
    def postprocess(self, inference_output):
        # TorchServe expects a list of strings as output
        return [json.dumps(inference_output)]
