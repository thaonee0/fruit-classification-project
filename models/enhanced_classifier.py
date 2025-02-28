import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import logging
import os

class FruitQualityClassifier:
    def __init__(self, model_path=None, device=None):
        """
        Khởi tạo classifier với model đã được huấn luyện
        
        Args:
            model_path (str): Đường dẫn đến file model (.pth)
            device (str): 'cuda' hoặc 'cpu'
        """
        # Thiết lập device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Định nghĩa các classes
        self.classes = ['Bad Quality', 'Good Quality', 'Mixed Quality']
        
        # Color mapping cho từng loại chất lượng
        self.color_map = {
            'Bad Quality_Fruits': (0, 0, 255),     # Đỏ
            'Good Quality_Fruits': (0, 255, 0),    # Xanh lá
            'Mixed Quality_Fruits': (0, 255, 255)  # Vàng
        }
        
        # Khởi tạo model
        try:
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.classes))
            
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logging.info(f"Loaded model from {model_path}")
            else:
                logging.warning("No model path provided or file not found. Using uninitialized model.")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise
        
        # Định nghĩa transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path):
        """
        Tiền xử lý ảnh đầu vào
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            
        Returns:
            torch.Tensor: Tensor ảnh đã được xử lý
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict_single_image(self, image_path, return_visualization=False):
        """
        Dự đoán chất lượng của một ảnh trái cây
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            return_visualization (bool): Nếu True, trả về cả ảnh đã được vẽ kết quả
            
        Returns:
            dict: Kết quả dự đoán với các thông tin:
                - quality: Loại chất lượng dự đoán
                - confidence: Độ tin cậy của dự đoán
                - probabilities: Xác suất cho từng loại chất lượng
                - visualization: (optional) Ảnh đã được vẽ kết quả
        """
        try:
            # Tiền xử lý ảnh
            image_tensor = self.preprocess_image(image_path)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # Lấy kết quả
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100
            
            # Tạo dictionary chứa xác suất của từng loại
            prob_dict = {
                class_name: prob.item() * 100 
                for class_name, prob in zip(self.classes, probabilities)
            }
            
            # Kết quả cơ bản
            result = {
                'quality': self.classes[predicted_class],
                'confidence': confidence,
                'probabilities': prob_dict
            }
            
            # Thêm visualization nếu được yêu cầu
            if return_visualization:
                result['visualization'] = self.visualize_prediction(image_path, result)
                
            return result
            
        except Exception as e:
            logging.error(f"Error predicting image: {str(e)}")
            raise

    def visualize_prediction(self, image_path, prediction_result):
        """
        Vẽ kết quả dự đoán lên ảnh
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            prediction_result (dict): Kết quả từ predict_single_image
            
        Returns:
            numpy.ndarray: Ảnh đã được vẽ kết quả
        """
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Cannot read image")
            
            # Lấy thông tin dự đoán
            quality = prediction_result['quality']
            confidence = prediction_result['confidence']
            color = self.color_map[quality]
            
            # Vẽ khung màu
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w-1, h-1), color, 3)
            
            # Chuẩn bị text
            text = f"{quality} ({confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Tính toán vị trí text
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = 10
            text_y = 30
            
            # Vẽ background cho text
            cv2.rectangle(image, 
                         (text_x-5, text_y-text_h-5),
                         (text_x+text_w+5, text_y+5),
                         color, -1)
            
            # Vẽ text
            cv2.putText(image, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Vẽ probability bar
            bar_height = 20
            bar_margin = 10
            total_height = (len(self.classes) * (bar_height + bar_margin)) + bar_margin
            
            # Vẽ probability bar cho từng class
            for i, (class_name, prob) in enumerate(prediction_result['probabilities'].items()):
                # Vị trí của bar
                bar_y = h - total_height + (i * (bar_height + bar_margin))
                bar_width = int((w - 20) * (prob / 100))
                
                # Vẽ background bar
                cv2.rectangle(image,
                            (10, bar_y),
                            (w-10, bar_y + bar_height),
                            (128, 128, 128), -1)
                
                # Vẽ probability bar
                cv2.rectangle(image,
                            (10, bar_y),
                            (10 + bar_width, bar_y + bar_height),
                            self.color_map[class_name], -1)
                
                # Thêm text
                text = f"{class_name}: {prob:.1f}%"
                cv2.putText(image, text,
                           (15, bar_y + bar_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
            
            return image
            
        except Exception as e:
            logging.error(f"Error visualizing prediction: {str(e)}")
            raise

    def predict_batch(self, image_paths, batch_size=32):
        """
        Dự đoán cho nhiều ảnh cùng lúc
        
        Args:
            image_paths (list): List các đường dẫn ảnh
            batch_size (int): Kích thước batch
            
        Returns:
            list: List các kết quả dự đoán
        """
        results = []
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_tensors = torch.stack([
                    self.preprocess_image(path).squeeze(0)
                    for path in batch_paths
                ])
                
                with torch.no_grad():
                    outputs = self.model(batch_tensors)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                for j, probs in enumerate(probabilities):
                    predicted_class = torch.argmax(probs).item()
                    confidence = probs[predicted_class].item() * 100
                    
                    prob_dict = {
                        class_name: prob.item() * 100
                        for class_name, prob in zip(self.classes, probs)
                    }
                    
                    results.append({
                        'image_path': batch_paths[j],
                        'quality': self.classes[predicted_class],
                        'confidence': confidence,
                        'probabilities': prob_dict
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error in batch prediction: {str(e)}")
            raise

    def get_quality_color(self, quality):
        """
        Trả về màu tương ứng với từng loại chất lượng
        
        Args:
            quality (str): Tên loại chất lượng
            
        Returns:
            tuple: Màu BGR tương ứng
        """
        return self.color_map.get(quality, (255, 255, 255))  # Mặc định là màu trắng