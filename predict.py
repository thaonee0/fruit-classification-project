import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Định nghĩa các tham số giống quá trình train
img_size = (224, 224)

isLoop = True


# Tải mô hình đã train
model_path = "final_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠ Không tìm thấy mô hình: {model_path}")

model = tf.keras.models.load_model(model_path)

# Danh sách nhãn (các loại quả)
class_names = sorted(os.listdir("D:/fruit-classification-project/CROPPED_DATA"))

def preprocess_image(img_path):
    """ Tiền xử lý ảnh trước khi đưa vào mô hình dự đoán """
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    img_array /= 255.0  # Chuẩn hóa ảnh
    return img_array

def predict_image(img_path):
    """ Dự đoán loại quả từ ảnh """
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)

    # Tìm nhãn có xác suất cao nhất
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Độ tin cậy %

    return class_names[predicted_class], confidence

def select_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        predicted_label, confidence = predict_image(file_path)

        # Hiển thị ảnh và kết quả dự đoán
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Dự đoán: {predicted_label}\nĐộ tin cậy: {confidence:.2f}%")
        plt.show()

    else:
        print("⚠ Không có ảnh nào được chọn!")
# Giao diện chọn file ảnh
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    while isLoop:
        select_and_predict()
        print("Bạn có muốn tiếp tục dự đoán không? (y/n)")
        ans = input()
        if ans.lower()!= 'y':
            isLoop = False
            break
    print("Kết thúc chương trình")
    root.mainloop
