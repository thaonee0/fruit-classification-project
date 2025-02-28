import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
from models.detector import FruitDetector 
from trained.train_mobilenetv2 import train_model
import cv2
import numpy as np
import threading
from datetime import datetime

class ImageTestDialog:
    # [Previous ImageTestDialog class implementation remains the same]
    pass

class FruitClassification:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classification")
        
        # Main container
        main_container = ttk.Frame(root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(main_container, text="Dataset Selection", padding="5")
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dataset path selection
        path_frame = ttk.Frame(dataset_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="Dataset Path:").pack(side=tk.LEFT)
        self.dataset_path = tk.StringVar()
        self.entry_dataset = ttk.Entry(path_frame, textvariable=self.dataset_path, width=50)
        self.entry_dataset.pack(side=tk.LEFT, padx=5)
        
        btn_browse = ttk.Button(path_frame, text="Browse", command=self.select_dataset)
        btn_browse.pack(side=tk.LEFT, padx=5)
        
        # Button frame
        button_frame = ttk.Frame(dataset_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Control buttons
        ttk.Button(button_frame, text="Detector", width=15, 
                  command=self.start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Train", width=15, 
                  command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test Image", width=15, 
                  command=self.test_single_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Log", width=15, 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Status and log frame
        log_frame = ttk.LabelFrame(main_container, text="Status Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrolled text widget for logging
        self.results_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize other necessary variables
        self.train_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.detector = None
        self.dataset_info = {}  # Store dataset information
        
        # Log initial message
        self.log_message("System initialized and ready.")
        
    def log_message(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()

    def analyze_dataset_structure(self, path):
        """Analyze dataset structure and return detailed information"""
        dataset_info = {
            'total_images': 0,
            'classes': {},
            'is_valid': False,
            'structure_type': None  # 'flat' or 'hierarchical'
        }
        
        try:
            # Check if path exists
            if not os.path.exists(path):
                return dataset_info
            
            # Look for class directories
            contents = os.listdir(path)
            potential_class_dirs = [d for d in contents if os.path.isdir(os.path.join(path, d))]
            
            # First check for hierarchical structure (class directories)
            if potential_class_dirs:
                dataset_info['structure_type'] = 'hierarchical'
                for class_dir in potential_class_dirs:
                    class_path = os.path.join(path, class_dir)
                    image_count = sum(1 for f in os.listdir(class_path) 
                                    if os.path.isfile(os.path.join(class_path, f)) and 
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                    if image_count > 0:
                        dataset_info['classes'][class_dir] = image_count
                        dataset_info['total_images'] += image_count
            
            # If no valid class directories found, check for flat structure
            if dataset_info['total_images'] == 0:
                dataset_info['structure_type'] = 'flat'
                image_count = sum(1 for f in contents 
                                if os.path.isfile(os.path.join(path, f)) and 
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                if image_count > 0:
                    dataset_info['classes']['default'] = image_count
                    dataset_info['total_images'] = image_count
            
            dataset_info['is_valid'] = dataset_info['total_images'] > 0
            
        except Exception as e:
            self.log_message(f"Error analyzing dataset: {str(e)}")
            
        return dataset_info

    def select_dataset(self):
        """Select and validate dataset directory"""
        selected_path = filedialog.askdirectory(title="Select Dataset Directory")
        if selected_path:
            self.dataset_path.set(selected_path)
            
            # Analyze dataset structure
            self.dataset_info = self.analyze_dataset_structure(selected_path)
            
            if self.dataset_info['is_valid']:
                self.log_message(f"Dataset selected: {selected_path}")
                self.log_message(f"Dataset structure: {self.dataset_info['structure_type']}")
                self.log_message(f"Total images found: {self.dataset_info['total_images']}")
                
                # Log class distribution if hierarchical
                if self.dataset_info['structure_type'] == 'hierarchical':
                    self.log_message("Class distribution:")
                    for class_name, count in self.dataset_info['classes'].items():
                        self.log_message(f"  - {class_name}: {count} images")
            else:
                self.log_message("⚠️ Warning: No valid images found in the selected directory")
                messagebox.showwarning("Warning", 
                    "No valid images found in the selected directory.\n"
                    "Please ensure the directory contains image files in the correct structure.")

    def start_training(self):
        """Start the training process with validated dataset"""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset directory first!")
            return
            
        if not self.dataset_info.get('is_valid', False):
            messagebox.showerror("Error", 
                "Invalid dataset structure.\n"
                "Please ensure your dataset contains valid image files in the correct structure.")
            return
            
        if self.dataset_info['total_images'] == 0:
            messagebox.showerror("Error", "No images found in the dataset!")
            return
            
        self.log_message("Starting training process...")
        self.log_message(f"Training with {self.dataset_info['total_images']} images")
        
        def training_thread():
            try:
                train_model(self.dataset_path.get())
                self.log_message("✅ Training completed successfully!")
                messagebox.showinfo("Success", "Model training completed!")
            except Exception as e:
                error_msg = f"❌ Training error: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
        
        thread = threading.Thread(target=training_thread)
        thread.start()

    def start_detection(self):
        """Start detection with validated dataset"""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset directory first!")
            return
            
        if not self.dataset_info.get('is_valid', False):
            messagebox.showerror("Error", "Invalid dataset structure!")
            return
            
        try:
            self.log_message("Initializing detector...")
            self.detector = FruitDetector(
                num_classes=len(self.dataset_info['classes']),
                train_data_path=self.dataset_path.get()
            )
            
            self.log_message("Processing images...")
            processed_images = self.detector.process_training_folder()
            
            self.log_message(f"✅ Detection completed. Processed {len(processed_images)} images")
            self.log_message(f"Results saved in: {self.detector.output_dir}")
            
            messagebox.showinfo("Success", "Detection process completed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Detection error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)

    def test_single_image(self):
        """Test single image with initialized detector"""
        if not self.detector:
            messagebox.showerror("Error", "Please run detection first!")
            return
        dialog = ImageTestDialog(self.root, self.detector)
        self.root.wait_window(dialog.dialog)

    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.log_message("Log cleared.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassification(root)
    root.mainloop()