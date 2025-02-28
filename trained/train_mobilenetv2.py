import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import UnidentifiedImageError
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime

# Định nghĩa các hằng số
DATASET_PATH = r"D:\2025\1.DO AN TOT NGHIEP\fruit-classification-project\detection_results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Định nghĩa đường dẫn lưu model
MODEL_DIR = "trained_models"
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_quality_model.pth")

# Tạo thư mục lưu model nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# Chuyển đổi dữ liệu
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["Bad Quality", "Good Quality", "Mixed Quality"]
        self.image_paths = []
        self.labels = []
        self.class_counts = {cls: 0 for cls in self.classes}  # Đếm số lượng ảnh mỗi lớp

        # Kiểm tra xem các thư mục lớp tồn tại
        missing_classes = []
        for class_name in self.classes:
            if not os.path.exists(os.path.join(root_dir, class_name)):
                missing_classes.append(class_name)
        
        if missing_classes:
            raise ValueError(f"Không tìm thấy các thư mục lớp sau: {', '.join(missing_classes)}")

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    # Kiểm tra nếu là thư mục thì bỏ qua
                    if os.path.isdir(img_path):
                        continue
                    try:
                        with Image.open(img_path) as img:  # Kiểm tra nếu là ảnh hợp lệ
                            self.image_paths.append(img_path)
                            self.labels.append(label)
                            self.class_counts[class_name] += 1
                    except (UnidentifiedImageError, PermissionError) as e:
                        print(f"Bỏ qua tệp không hợp lệ {img_path}: {str(e)}")

        if len(self.image_paths) == 0:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong dataset!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(dataset_path):
    """Huấn luyện mô hình phân loại chất lượng trái cây"""
    
    print(f"\n{'='*50}")
    print(f"Bắt đầu quá trình huấn luyện: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    if not os.path.exists(dataset_path):
        raise ValueError(f"❌ Thư mục dataset không tồn tại: {dataset_path}")

    print(f"📂 Đang sử dụng dataset từ: {dataset_path}")

    try:
        # Load dataset
        full_dataset = FruitDataset(dataset_path, transform=transform)
        print(f"✅ Tổng số ảnh trong dataset: {len(full_dataset)}")
        
        # In thông tin về số lượng ảnh trong mỗi lớp
        print("\nPhân bố ảnh trong các lớp:")
        for class_name, count in full_dataset.class_counts.items():
            print(f"   🔹 {class_name}: {count} ảnh")

        # Chia dataset thành training và validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        print(f"\nChia dataset:")
        print(f"   🔹 Tập training: {train_size} ảnh")
        print(f"   🔹 Tập validation: {val_size} ảnh")

        # Tạo data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # Load mô hình MobileNetV2
        num_classes = len(full_dataset.classes)
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        # Kiểm tra và load model cũ nếu có
        if os.path.exists(MODEL_PATH):
            print(f"\n🔄 Đang tiếp tục train từ model cũ: {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH))
        else:
            print("\n🚀 Training từ đầu vì không tìm thấy model cũ.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n💻 Sử dụng thiết bị: {device}")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        # Huấn luyện với early stopping
        best_val_acc = 0
        patience = 10
        patience_counter = 0

        print("\nBắt đầu huấn luyện...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total

            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            scheduler.step(val_acc)

            # Lưu model nếu tốt hơn
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"💾 Đã lưu model mới (accuracy: {val_acc:.2f}%)")
            else:
                patience_counter += 1

            # Dừng train nếu không cải thiện sau nhiều epochs
            if patience_counter >= patience:
                print(f"\n⏹️ Dừng sớm sau {epoch+1} epochs do không cải thiện")
                break

            print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")
            print(f"   🔹 Train Loss: {running_loss/len(train_loader):.4f}")
            print(f"   🔹 Train Accuracy: {train_acc:.2f}%")
            print(f"   🔹 Validation Accuracy: {val_acc:.2f}%")

        print("\n✅ Hoàn thành huấn luyện!")
        print(f"📊 Accuracy tốt nhất trên tập validation: {best_val_acc:.2f}%")
        print(f"💾 Model đã được lưu tại: {MODEL_PATH}")

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        raise

if __name__ == "__main__":
    train_model(DATASET_PATH)