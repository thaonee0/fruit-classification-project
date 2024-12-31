import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from models.detector import FruitDetector
import cv2
import numpy as np

class ImageTestDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Test Image")
        self.dialog.geometry("600x400")
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Image information frame
        info_frame = ttk.Frame(self.dialog, padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="Image Path:").pack(side=tk.LEFT)
        self.image_path = tk.StringVar()
        self.path_entry = ttk.Entry(info_frame, textvariable=self.image_path, width=50)
        self.path_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(info_frame, text="Browse", command=self.browse_image).pack(side=tk.LEFT)
        
        # Image preview frame
        self.preview_frame = ttk.Frame(self.dialog, padding="10")
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image preview
        self.canvas = tk.Canvas(self.preview_frame, width=500, height=300)
        self.canvas.pack(pady=10)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog, padding="10")
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        
        self.result = None
        self.photo = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_path.set(file_path)
            self.show_preview(file_path)

    def show_preview(self, image_path):
        try:
            # Clear previous image
            self.canvas.delete("all")
            
            # Load and resize image for preview
            image = Image.open(image_path)
            # Calculate resize ratio while maintaining aspect ratio
            display_size = (500, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and keep reference
            self.photo = ImageTk.PhotoImage(image)
            
            # Center image in canvas
            x = (500 - self.photo.width()) // 2
            y = (300 - self.photo.height()) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def ok_clicked(self):
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image file")
            return
        self.result = self.image_path.get()
        self.dialog.destroy()

class FruitClassification:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classification")
        
        # Configure main frame with gray background
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame['style'] = 'Gray.TFrame'
        
        # Create and configure style
        style = ttk.Style()
        style.configure('Gray.TFrame', background='#f0f0f0')
        style.configure('Gray.TLabelframe', background='#f0f0f0')
        style.configure('Gray.TLabelframe.Label', background='#f0f0f0')
        
        # Training data section
        ttk.Label(main_frame, text="Training data folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_path = tk.StringVar(value="")
        ttk.Entry(main_frame, textvariable=self.train_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_train).grid(row=0, column=2)

        # Test data section
        ttk.Label(main_frame, text="Test data folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_path = tk.StringVar(value="")
        ttk.Entry(main_frame, textvariable=self.test_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_test).grid(row=1, column=2)
        ttk.Button(main_frame, text="Close", command=root.destroy).grid(row=1, column=3)
        
        # Training function section
        train_frame = ttk.LabelFrame(main_frame, text="Training function", padding="15", style='Gray.TLabelframe')
        train_frame.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=10, padx=50)
        
        # Center the content in train_frame
        train_frame.grid_columnconfigure(0, weight=1)
        train_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(train_frame, text="Epoch:", font=('Arial', 10)).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.epoch = tk.StringVar(value="1000")
        ttk.Entry(train_frame, textvariable=self.epoch, width=15).grid(row=0, column=1, sticky="w", pady=5)
        
        ttk.Label(train_frame, text="Batch size:", font=('Arial', 10)).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.batch_size = tk.StringVar(value="100")
        ttk.Entry(train_frame, textvariable=self.batch_size, width=15).grid(row=1, column=1, sticky="w", pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(train_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        train_button = ttk.Button(buttons_frame, text="Training", width=15, command=self.start_training)
        train_button.pack(side=tk.LEFT, padx=5)
        
        detect_button = ttk.Button(buttons_frame, text="Detector", width=15, command=self.start_detection)
        detect_button.pack(side=tk.LEFT, padx=5)

        # Testing function section
        test_frame = ttk.LabelFrame(main_frame, text="Testing function", padding="15", style='Gray.TLabelframe')
        test_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=10, padx=50)
        
        # Center the buttons in test_frame
        button_frame = ttk.Frame(test_frame, style='Gray.TFrame')
        button_frame.grid(row=0, column=0, pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(button_frame, text="Test database", width=15, command=self.test_database).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Test an image", width=15, command=self.test_single_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Test Images", width=15, command=self.test_images).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Clear", width=15, command=self.clear_results).grid(row=0, column=3, padx=5)
        
        # Results area
        self.results_text = tk.Text(main_frame, height=15, width=70)
        self.results_text.grid(row=4, column=0, columnspan=4, pady=10)

        self.detector = None  # Add this line to store detector instance

    def browse_train(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.train_path.set(folder_path)

    def browse_test(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.test_path.set(folder_path)

    def count_images(self, folder_path):
        image_count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_count += 1
        return image_count

    def start_training(self):
        if not self.train_path.get():
            messagebox.showerror("Error", "Please select training data folder")
            return
        # Add your training logic here
        pass

    def start_detection(self):
        try:
            if not self.train_path.get():
                messagebox.showerror("Error", "Please select a training data folder first.")
                return

            self.detector = FruitDetector(
                num_classes=2,  # Adjust as needed
                train_data_path=self.train_path.get()
            )

            self.results_text.insert(tk.END, "\nProcessing training data folder...\n")
            self.root.update()

            detections = self.detector.process_training_folder(save_results=True)

            total_images = len(detections)
            detected_fruits = sum(len(d['detections']) for d in detections)

            results_text = f"\nDetection Results:\n"
            results_text += f"- Total images processed: {total_images}\n"
            results_text += f"- Total fruits detected: {detected_fruits}\n"

            self.results_text.insert(tk.END, results_text)
            messagebox.showinfo("Success", "Detection process completed successfully!")

        except Exception as e:
            error_message = f"Error during detection: {str(e)}"
            messagebox.showerror("Error", error_message)
            self.results_text.insert(tk.END, f"\n{error_message}\n")

    def test_database(self):
        if not self.test_path.get():
            messagebox.showerror("Error", "Please select test data folder")
            return
        # Add your database testing logic here
        pass

    def test_single_image(self):
        dialog = ImageTestDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            # Here you can add your image classification logic
            self.results_text.insert(tk.END, f"\nTesting image: {dialog.result}")

    def test_images(self):
        if not self.train_path.get() or not self.test_path.get():
            messagebox.showerror("Error", "Please select both training and test data folders")
            return
            
        # Count images in training folder
        train_count = self.count_images(self.train_path.get())
        self.results_text.insert(tk.END, f"\nTraining dataset: Found {train_count} images")
        
        # Count images in test folder
        test_count = self.count_images(self.test_path.get())
        self.results_text.insert(tk.END, f"\nTest dataset: Found {test_count} images")

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassification(root)
    root.mainloop()
