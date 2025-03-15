import os
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Load mô hình YOLOv8 pre-trained
model = YOLO('yolov8n.pt')

def preprocess_image(image_path):
    """ Tiền xử lý ảnh """
    img = cv2.imread(image_path)
    return img if img is not None else None

def detect_and_crop(image_path, output_base_dir, relative_path, min_area=500, target_size=(224, 224)):
    """ Detect vật thể, crop vùng chứa và lưu ảnh đã resize. """
    if not os.path.exists(image_path):
        print(f"⚠ Ảnh không tồn tại: {image_path}")
        return None

    image = preprocess_image(image_path)
    if image is None:
        print(f"⚠ Không thể đọc ảnh: {image_path}")
        return None

    # Chạy YOLO detect
    results = model(image, conf=0.7, iou=0.5, agnostic_nms=True, verbose=False)
    if not results or len(results[0].boxes) == 0:
        print(f"🚫 Không phát hiện vật thể: {image_path}")
        return None

    # Tạo thư mục lưu ảnh
    class_output_dir = os.path.join(output_base_dir, os.path.dirname(relative_path))
    os.makedirs(class_output_dir, exist_ok=True)

    saved_paths = []
    
    for i, box in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, cls = map(int, box[:6])

        # Giới hạn bounding box trong phạm vi ảnh
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

        # Kiểm tra bounding box hợp lệ
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or (x2 - x1) <= 0 or (y2 - y1) <= 0:
            print(f"⚠ Bounding box lỗi, bỏ qua: {image_path}")
            continue

        # Cắt và resize ảnh
        cropped_img = image[y1:y2, x1:x2]
        cropped_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)

        # Lưu ảnh cropped
        filename = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(class_output_dir, f"{filename}_crop{i}.jpg")
        cv2.imwrite(save_path, cropped_img)
        saved_paths.append(save_path)

    return saved_paths if saved_paths else None

def process_dataset(input_dir, output_base_dir):
    """ Duyệt qua dataset và crop các vật thể phát hiện được. """
    os.makedirs(output_base_dir, exist_ok=True)

    new_filepaths, labels = [], []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(img_path, input_dir)
                cropped_paths = detect_and_crop(img_path, output_base_dir, relative_path)

                if cropped_paths:
                    for cropped_path in cropped_paths:
                        new_filepaths.append(cropped_path)
                        labels.append(os.path.basename(os.path.dirname(relative_path)))  # Chỉ lấy thư mục cha trực tiếp

    df = pd.DataFrame({'filepaths': new_filepaths, 'labels': labels})
    return df

if __name__ == "__main__":
    input_dir = r"D:\fruit-classification-project\DATA\Good Quality_Fruits"
    output_dir = r"D:\fruit-classification-project\CROPPED_DATA"

    df = process_dataset(input_dir, output_dir)
    print(f"✅ Xử lý xong! Tổng số ảnh cropped: {len(df)}")