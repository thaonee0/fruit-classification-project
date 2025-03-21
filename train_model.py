import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Kiểm tra GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Cấu hình mô hình
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)
batch_size = 8  
epochs = 30
ask_epoch = 3  # Epoch đầu tiên sẽ hỏi người dùng

def load_data_from_folder(base_dir):
    """
    Đọc dữ liệu từ thư mục ảnh và gán nhãn dựa trên tên thư mục.
    """
    image_paths = []
    labels = []
    
    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(label_path, img_file))
                    labels.append(label)
    
    df = pd.DataFrame({'filepaths': image_paths, 'labels': labels})
    print(f"📌 Tìm thấy {len(df)} ảnh từ thư mục {base_dir}")
    return df

# Load dataset
base_dir = "D:/fruit-classification-project/CROPPED_DATA"
df = load_data_from_folder(base_dir)

# Chia dữ liệu train, validation, test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=42)

# Định nghĩa số lớp
class_names = sorted(train_df['labels'].unique())
class_count = len(class_names)

# Tạo ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = datagen.flow_from_dataframe(
    train_df, x_col="filepaths", y_col="labels",
    target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

valid_gen = datagen.flow_from_dataframe(
    valid_df, x_col="filepaths", y_col="labels",
    target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

test_gen = datagen.flow_from_dataframe(
    test_df, x_col="filepaths", y_col="labels",
    target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
)

# Load EfficientNetB5
base_model = tf.keras.applications.EfficientNetB5(
    include_top=False, weights="imagenet", input_shape=img_shape, pooling='max'
)

base_model.trainable = True

# Thêm các lớp fully connected
x = base_model.output
x = BatchNormalization()(x)
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.016))(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.016))(x)
x = Dropout(0.45)(x)
output = Dense(class_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback ASK để kiểm soát training
class ASK(keras.callbacks.Callback):
    def __init__(self, epochs, ask_epoch):
        super(ASK, self).__init__()
        self.ask_epoch = ask_epoch
        self.epochs = epochs
        self.ask = True
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        tr_duration = time.time() - self.start_time
        print(f'⏳ Training hoàn tất sau {tr_duration // 60:.1f} phút')

    def on_epoch_end(self, epoch, logs=None):
        if self.ask and (epoch + 1 == self.ask_epoch):
            ans = input('\n🔹 Nhập số epoch tiếp theo hoặc "H" để dừng: ')
            if ans.lower() == 'h':
                print(f'🛑 Training dừng tại epoch {epoch + 1}')
                self.model.stop_training = True
            else:
                self.ask_epoch += int(ans)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ASK(epochs, ask_epoch)
]

# Huấn luyện mô hình
history = model.fit(
    train_gen, epochs=epochs, validation_data=valid_gen,
    verbose=1, callbacks=callbacks, shuffle=False
)

# Đánh giá trên tập test
test_loss, test_acc = model.evaluate(test_gen)
print(f"🎯 Độ chính xác trên tập kiểm tra: {test_acc * 100:.2f}%")

# Lưu mô hình
model.save("final_model.h5")
print("✅ Mô hình đã được lưu!")


#vẽ biểu đồ
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """Vẽ biểu đồ Loss và Accuracy từ history của quá trình huấn luyện"""
    
    # Lấy dữ liệu từ history
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    # Tìm epoch có độ chính xác cao nhất và loss thấp nhất
    best_val_acc_epoch = np.argmax(val_acc) + 1
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_acc = val_acc[best_val_acc_epoch - 1]
    best_val_loss = val_loss[best_val_loss_epoch - 1]

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'g', label="Validation Loss")
    plt.scatter(best_val_loss_epoch, best_val_loss, s=150, c='blue', label=f'Best Loss Epoch: {best_val_loss_epoch}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    
    # Vẽ biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'g', label="Validation Accuracy")
    plt.scatter(best_val_acc_epoch, best_val_acc, s=150, c='blue', label=f'Best Acc Epoch: {best_val_acc_epoch}')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Gọi hàm để vẽ biểu đồ sau khi huấn luyện
plot_training_history(history)


