import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

class FruitClassification:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classification - NHÓM 1")
        
        # Training data section
        frame1 = ttk.Frame(root, padding="5")
        frame1.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(frame1, text="Training data folder:").grid(row=0, column=0, sticky=tk.W)
        self.train_path = tk.StringVar(value="Hãy tìm ảnh !!!")
        ttk.Entry(frame1, textvariable=self.train_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(frame1, text="Browse...", command=self.browse_train).grid(row=0, column=2)
        
        # Test data section
        frame2 = ttk.Frame(root, padding="5")
        frame2.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(frame2, text="Test data folder:").grid(row=0, column=0, sticky=tk.W)
        self.test_path = tk.StringVar(value="Hãy tìm ảnh !!!")
        ttk.Entry(frame2, textvariable=self.test_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(frame2, text="Browse...", command=self.browse_test).grid(row=0, column=2)
        
        # Training function section
        train_frame = ttk.LabelFrame(root, text="Training function", padding="5")
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Label(train_frame, text="Epoch:").grid(row=0, column=0, sticky=tk.W)
        self.epoch = tk.StringVar(value="1000")
        ttk.Entry(train_frame, textvariable=self.epoch, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(train_frame, text="Batch size:").grid(row=1, column=0, sticky=tk.W)
        self.batch_size = tk.StringVar(value="100")
        ttk.Entry(train_frame, textvariable=self.batch_size, width=10).grid(row=1, column=1, sticky=tk.W)
        
        train_button = ttk.Button(train_frame, text="Training", width=15)
        train_button.grid(row=2, column=0, columnspan=2, pady=5)
        train_frame.grid_columnconfigure(0, weight=1)
        train_frame.grid_columnconfigure(1, weight=1)
        
        # Testing function section
        test_frame = ttk.LabelFrame(root, text="Testing function", padding="5")
        test_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        button_frame = ttk.Frame(test_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Test database").grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Test an image").grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="Test Images").grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="Clear", command=self.clear_results).grid(row=0, column=3, padx=2)
        ttk.Button(button_frame, text="Close", command=root.quit).grid(row=0, column=4, padx=2)
        
        # Results display area
        self.results_text = tk.Text(root, height=15, width=60)
        self.results_text.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
    def browse_train(self):
        folder = filedialog.askdirectory()
        if folder:
            self.train_path.set(folder)
            
    def browse_test(self):
        folder = filedialog.askdirectory()
        if folder:
            self.test_path.set(folder)
    
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassification(root)
    root.mainloop()