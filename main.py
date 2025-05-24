import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

class ImageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier (EfficientNet)")
        self.root.geometry("700x650")
        self.root.minsize(600, 550)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        self.style.configure('Result.TLabel', font=('Helvetica', 11), foreground='#333')
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Accent.TButton', foreground='black', background='#e0e0e0',
                           font=('Helvetica', 10, 'bold'), padding=8, borderwidth=1, relief='raised')
        self.style.configure('ImageFrame.TFrame', background='white')
        
        # Load EfficientNet model (more accurate than MobileNet)
        self.model = EfficientNetB0(weights='imagenet')
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="AI Image Classifier (EfficientNet)", style='Title.TLabel').pack(side=tk.LEFT)
        
        # Image selection
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=10)
        self.select_btn = ttk.Button(select_frame, text="Select Image", command=self.select_image, style='Accent.TButton')
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.file_label = ttk.Label(select_frame, text="No image selected", font=('Helvetica', 9), foreground='#555555')
        self.file_label.pack(side=tk.LEFT)
        
        # Image display
        self.image_frame = ttk.Frame(main_frame, relief=tk.GROOVE, borderwidth=2, style='ImageFrame.TFrame')
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.image_canvas = tk.Canvas(self.image_frame, bg='white', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.result_label = ttk.Label(results_frame, text="Select an image to see classification results", 
                                   style='Result.TLabel', justify=tk.LEFT, wraplength=350, background='white', padding=10)
        self.result_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, 
                             padding=(5,2), font=('Helvetica', 9), background='#f0f0f0')
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            messagebox.showerror("Unsupported Format", f"File format {file_ext} is not supported.")
            return
            
        self.file_label.config(text=os.path.basename(file_path))
        self.status_var.set("Processing image...")
        self.root.update_idletasks()
        
        try:
            decoded, img = self.process_image(file_path)
            self.display_image(img)
            
            result_text = "\n".join([f"{i+1}. {label.title()} ({prob*100:.2f}%)" 
                                   for i, (_, label, prob) in enumerate(decoded)])
            self.result_label.config(text=result_text)
            self.status_var.set("Classification complete")
            
        except Exception as e:
            self.status_var.set("Error processing image")
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
    
    def process_image(self, img_path):
        # Load and preprocess for EfficientNet
        img = keras_image.load_img(img_path, target_size=(224, 224))  # EfficientNetB0 expects 224x224
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = effnet_preprocess(x)  # EfficientNet-specific preprocessing
        preds = self.model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
        original_img = Image.open(img_path)
        return decoded, original_img
    
    def display_image(self, img):
        self.image_canvas.delete("all")
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Maintain aspect ratio
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if canvas_ratio > img_ratio:
            new_height = canvas_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / img_ratio)
        
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        
        # Center the image
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        self.image_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)
        self.image_canvas.bind("<Configure>", lambda e: self.on_canvas_resize(img))
    
    def on_canvas_resize(self, original_img):
        if hasattr(self, 'tk_image'):
            self.display_image(original_img)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognitionApp(root)
    app.run()