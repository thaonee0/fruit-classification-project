import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
import os

class FruitQualityPredictor:
    def __init__(self, model_path="trained_models/fruit_quality_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.classes = ['Bad Quality', 'Good Quality', 'Mixed Quality']

    def load_model(self, model_path):
        try:
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.last_channel, 3)  # 3 classes
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        try:
            # Load và transform ảnh
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item() * 100
                
            # Trả về kết quả
            return {
                'class': self.classes[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    cls: prob.item() * 100 
                    for cls, prob in zip(self.classes, probabilities[0])
                }
            }
        except Exception as e:
            raise Exception(f"Error predicting image: {str(e)}")

    def process_and_draw(self, image_path):
        try:
            # Đọc ảnh gốc
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise Exception("Could not read image")

            # Thực hiện dự đoán
            result = self.predict_image(image_path)
            
            # Vẽ kết quả lên ảnh
            img_with_text = original_image.copy()
            
            # Thêm một hộp thông tin ở góc trên trái
            margin = 10
            text_color = (255, 255, 255)  # Màu trắng
            bg_color = (0, 0, 0)  # Màu đen
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Chuẩn bị các dòng text
            lines = [
                f"Class: {result['class']}",
                f"Confidence: {result['confidence']:.2f}%"
            ]
            
            # Tính toán kích thước của hộp thông tin
            text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] 
                         for line in lines]
            box_width = max(size[0] for size in text_sizes) + 2 * margin
            line_height = max(size[1] for size in text_sizes)
            box_height = (line_height + margin) * len(lines) + margin
            
            # Vẽ hộp đen làm nền
            cv2.rectangle(img_with_text, 
                         (0, 0), 
                         (box_width, box_height), 
                         bg_color, 
                         -1)
            
            # Vẽ text
            y = margin + line_height
            for line in lines:
                cv2.putText(img_with_text, 
                           line, 
                           (margin, y), 
                           font, 
                           font_scale, 
                           text_color, 
                           thickness)
                y += line_height + margin

            return img_with_text
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")