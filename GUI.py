import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

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
        self.train_path = tk.StringVar(value="Fruit Recognition/fruits-360/")
        ttk.Entry(main_frame, textvariable=self.train_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse...").grid(row=0, column=2)
        
        # Test data section
        ttk.Label(main_frame, text="Test data folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_path = tk.StringVar(value="Fruit Recognition/fruits-360/")
        ttk.Entry(main_frame, textvariable=self.test_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse...").grid(row=1, column=2)
        ttk.Button(main_frame, text="Close").grid(row=1, column=3)
        
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
        
        train_button = ttk.Button(train_frame, text="Training", width=20)
        train_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Testing function section
        test_frame = ttk.LabelFrame(main_frame, text="Testing function", padding="15", style='Gray.TLabelframe')
        test_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=10, padx=50)
        
        # Center the buttons in test_frame
        button_frame = ttk.Frame(test_frame, style='Gray.TFrame')
        button_frame.grid(row=0, column=0, pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(button_frame, text="Test database", width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Test an image", width=15).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Test Images", width=15).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Clear", width=15).grid(row=0, column=3, padx=5)
        
        # Results area
        self.results_text = tk.Text(main_frame, height=15, width=70)
        self.results_text.grid(row=4, column=0, columnspan=4, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassification(root)
    root.mainloop()