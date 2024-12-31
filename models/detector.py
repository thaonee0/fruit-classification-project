import torch
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
import logging

class FruitDetector:
    def __init__(self, num_classes, yolo_model='yolov8n.pt', train_data_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_model)
        self.train_data_path = train_data_path
        self.num_classes = num_classes
        self.model_save_dir = "trained_models"
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Tắt logging của YOLO để không in thông tin thừa
        logging.getLogger('ultralytics').setLevel(logging.ERROR)  # Tắt các thông báo của YOLO

    def process_training_folder(self, save_results=True):
        if not self.train_data_path or not os.path.exists(self.train_data_path):
            raise ValueError("Training data path is not provided or does not exist.")

        all_detections = []
        for root, _, files in os.walk(self.train_data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    detections = self.detect_and_classify(image_path)
                    result = {
                        'image_path': image_path,
                        'detections': detections
                    }
                    all_detections.append(result)

        if save_results and all_detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.model_save_dir, f"detection_results_{timestamp}.json")
            with open(results_path, 'w') as f:
                json.dump(all_detections, f, indent=4)

        return all_detections

    def detect_and_classify(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        image_resized = cv2.resize(image, (256, 192))

        # Chuyển đổi hình ảnh thành RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Sử dụng YOLO để phát hiện đối tượng trong hình ảnh mà không in thông tin thừa
        results = self.yolo(image_rgb, verbose=False)  # Tắt thông báo

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Chỉ lưu lại thông tin cần thiết (bounding box và confidence)
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf
                })

        return detections
