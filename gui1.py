import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import os
from models.model_tester import FruitQualityPredictor

# ĐƯỜNG DẪN CỐ ĐỊNH
FOLDER_PATH = r"D:\2025\1.DO AN TOT NGHIEP\fruit-classification-project\Validation"

class FruitQualityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Quality Detector")
        
        # Model
        self.predictor = FruitQualityPredictor()
        
        # Label & Buttons
        self.label = Label(root, text="Choose an image to classify")
        self.label.pack()
        
        self.btn_select = Button(root, text="Select Image", command=self.load_image)
        self.btn_select.pack()
        
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()
        
        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir=FOLDER_PATH, filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        # CHUẨN HÓA ĐƯỜNG DẪN
        file_path = os.path.abspath(file_path)  # Chuyển đường dẫn thành dạng chuẩn

        # KIỂM TRA FILE TỒN TẠI
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected image does not exist!")
            return

        try:
            # KIỂM TRA OPENCV ĐỌC ĐƯỢC ẢNH KHÔNG
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "OpenCV could not read the image!")
                return

            # Dự đoán chất lượng trái cây
            result = self.predictor.predict_image(file_path)

            # Xử lý ảnh và vẽ kết quả
            processed_image = self.predictor.process_and_draw(file_path)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(processed_image)

            # Hiển thị ảnh lên GUI
            self.img_tk = ImageTk.PhotoImage(image_pil)
            self.canvas.config(width=image_pil.width, height=image_pil.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            # Hiển thị kết quả
            self.result_label.config(text=f"Class: {result['class']}\nConfidence: {result['confidence']:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitQualityGUI(root)
    root.mainloop()
