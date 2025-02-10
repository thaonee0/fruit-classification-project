import torch
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import logging

class FruitDetector:
    def __init__(self, num_classes, yolo_model='yolov8n.pt', train_data_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_model)
        self.train_data_path = train_data_path
        self.num_classes = num_classes
        self.output_dir = "detection_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tắt logging của YOLO
        logging.getLogger('ultralytics').setLevel(logging.ERROR)

    def process_training_folder(self):
        if not self.train_data_path or not os.path.exists(self.train_data_path):
            raise ValueError("Training data path is not provided or does not exist.")

        processed_images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = os.path.join(self.output_dir, f"results_{timestamp}")
        os.makedirs(result_folder, exist_ok=True)

        for root, _, files in os.walk(self.train_data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    output_path = os.path.join(result_folder, f"detected_{file}")
                    
                    # Thực hiện detection và vẽ bounding boxes
                    detected_image = self.detect_and_draw(image_path)
                    
                    if detected_image is not None:
                        # Lưu ảnh với bounding boxes
                        cv2.imwrite(output_path, detected_image)
                        processed_images.append({
                            'original_path': image_path,
                            'result_path': output_path
                        })

        return processed_images

    def detect_and_draw(self, image_path):
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Tạo bản sao để vẽ lên
        image_with_boxes = image.copy()
        
        # Resize ảnh về kích thước chuẩn
        image_resized = cv2.resize(image, (256, 192))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Thực hiện detection
        results = self.yolo(image_rgb, verbose=False)
        
        # Tỷ lệ scale để vẽ boxes lên ảnh gốc
        scale_x = image.shape[1] / 256
        scale_y = image.shape[0] / 192
        
        # Vẽ mỗi bounding box được phát hiện
        for r in results:
            for box in r.boxes:
                # Lấy tọa độ box và confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Scale lại tọa độ về kích thước ảnh gốc
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Vẽ bounding box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Thêm confidence score
                label = f'Confidence: {conf:.2f}'
                cv2.putText(image_with_boxes, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image_with_boxes