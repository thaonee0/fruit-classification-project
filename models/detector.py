import torch
from ultralytics import YOLO
import cv2
import torchvision.transforms as transforms
from .classifier import FruitClassifier

class FruitDetector:
    def __init__(self, num_classes, yolo_model='yolov8n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_model)
        self.classifier = FruitClassifier(num_classes).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def detect_and_classify(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo(image)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                fruit_region = image[int(y1):int(y2), int(x1):int(x2)]
                
                if fruit_region.size != 0:
                    fruit_tensor = self.transform(fruit_region).unsqueeze(0)
                    fruit_tensor = fruit_tensor.to(self.device)
                    
                    with torch.no_grad():
                        output = self.classifier(fruit_tensor)
                        pred = torch.argmax(output).item()
                    
                    detections.append({
                        'box': box.xyxy[0].cpu().numpy(),
                        'class': pred
                    })
        return detections