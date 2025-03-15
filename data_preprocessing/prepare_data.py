import os
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from augment_data import augment_image

def get_image_paths(base_dir):
    """
    Duyệt qua thư mục dataset và lấy danh sách ảnh với nhãn.
    """
    image_paths, labels = [], []
    if not os.path.exists(base_dir):
        raise ValueError(f"❌ Thư mục không tồn tại: {base_dir}")
    
    for quality_folder in os.listdir(base_dir):
        quality_path = os.path.join(base_dir, quality_folder)
        if os.path.isdir(quality_path):
            for fruit_folder in os.listdir(quality_path):
                fruit_path = os.path.join(quality_path, fruit_folder)
                if os.path.isdir(fruit_path):
                    for img_file in os.listdir(fruit_path):
                        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            image_paths.append(os.path.join(fruit_path, img_file))
                            labels.append(fruit_folder)
    
    if len(image_paths) == 0:
        raise ValueError("❌ Không tìm thấy ảnh nào trong thư mục đầu vào!")
    return image_paths, labels

def balance_dataset(image_paths, labels, max_samples, output_dir, csv_path):
    """
    Cân bằng dataset bằng cách augment dữ liệu nếu cần và lưu vào CSV.
    """
    balanced_data = []
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = {label: labels.count(label) for label in set(labels)}
    class_dirs = {label: os.path.join(output_dir, label) for label in set(labels)}
    for class_dir in class_dirs.values():
        os.makedirs(class_dir, exist_ok=True)
    
    total_augmented = 0
    for label in set(labels):
        class_images = [img for img, lbl in zip(image_paths, labels) if lbl == label]
        num_images = len(class_images)
        
        if num_images < max_samples:
            delta = max_samples - num_images
            images_needed = class_images * (delta // num_images) + class_images[:(delta % num_images)]
            for idx, img_path in enumerate(images_needed):
                image = cv2.imread(img_path)
                if image is None:
                    continue
                aug_images = augment_image(image)
                for aug_idx, aug_img in enumerate(aug_images):
                    save_path = os.path.join(class_dirs[label], f"{label}_aug_{idx}_{aug_idx}.jpg")
                    cv2.imwrite(save_path, aug_img)
                    balanced_data.append([save_path, label])
                    total_augmented += 1
        
        for img in class_images[:max_samples]:
            save_path = os.path.join(class_dirs[label], os.path.basename(img))
            shutil.copy(img, save_path)
            balanced_data.append([save_path, label])
    
    print(f"✅ Augmented {total_augmented} ảnh để cân bằng dataset.")
    print(f"✅ Tổng số ảnh sau khi cân bằng: {len(balanced_data)}")
    
    # Lưu thông tin vào CSV
    df = pd.DataFrame(balanced_data, columns=['image_path', 'label'])
    df.to_csv(csv_path, index=False)
    print(f"✅ Lưu dữ liệu vào {csv_path}")
    
    return [row[0] for row in balanced_data], [row[1] for row in balanced_data]

def split_data(image_paths, labels, output_dir):
    """
    Chia dataset thành train, validation, test.
    """
    if len(set(labels)) == 1:
        raise ValueError("❌ Dataset chỉ có một lớp, không thể chia train/test!")
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    def save_images(paths, labels, subset_name):
        subset_dir = os.path.join(output_dir, subset_name)
        shutil.rmtree(subset_dir, ignore_errors=True)
        os.makedirs(subset_dir)
        for path, label in zip(paths, labels):
            class_dir = os.path.join(subset_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(path, os.path.join(class_dir, os.path.basename(path)))
        print(f"✅ {subset_name}: {len(paths)} ảnh")
    
    save_images(train_paths, train_labels, "train")
    save_images(valid_paths, valid_labels, "valid")
    save_images(test_paths, test_labels, "test")

if __name__ == "__main__":
    input_dir = r"D:\fruit-classification-project\CROPPED_DATA"
    output_dir = r"D:\fruit-classification-project\BALANCED_DATA"
    csv_path = r"D:\fruit-classification-project\balanced_data.csv"
    
    image_paths, labels = get_image_paths(input_dir)
    print(f"📌 Tìm thấy {len(image_paths)} ảnh trong dataset ban đầu.")
    
    balanced_paths, balanced_labels = balance_dataset(image_paths, labels, max_samples=200, output_dir=output_dir, csv_path=csv_path)
    split_data(balanced_paths, balanced_labels, output_dir)