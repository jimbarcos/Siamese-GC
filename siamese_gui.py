#!/usr/bin/env python3
"""
🐾 Siamese GC Capsule Network - GUI Application
A comprehensive GUI for analyzing animal image similarities using the trained Siamese model.

Features:
- Load trained model (.pth files)
- Import and compare two images
- Detailed similarity analysis with confidence scores
- Visual comparison and technical metrics
- Comprehensive reporting
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import json
from datetime import datetime


class ImprovedSiameseGCCapsuleNet(nn.Module):
    """Enhanced Siamese GC Capsule Network - Same as training notebook"""
    
    def __init__(self, embedding_dim=128):
        super(ImprovedSiameseGCCapsuleNet, self).__init__()
        
        # Enhanced convolutional layers with proper initialization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Capsule-inspired layers
        self.capsule_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.capsule_bn = nn.BatchNorm2d(512)
        
        # Global pooling and embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        
        # Enhanced embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x):
        """Forward pass for one image"""
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Capsule-inspired layer
        x = F.relu(self.capsule_bn(self.capsule_conv(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout and embedding
        x = self.dropout(x)
        x = self.embedding(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, img1, img2):
        """Forward pass for both images"""
        embed1 = self.forward_one(img1)
        embed2 = self.forward_one(img2)
        return embed1, embed2


def improved_cosine_similarity(embed1, embed2, temperature=1.2):
    """Improved cosine similarity with temperature scaling"""
    # Calculate cosine similarity
    cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)
    
    # Apply temperature scaling and sigmoid for better range
    scaled_sim = torch.sigmoid(cosine_sim * temperature)
    
    return scaled_sim


class SiameseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Siamese GC Capsule Network - Animal Similarity Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = 0.5  # Default threshold
        self.image1_path = None
        self.image2_path = None
        self.model_path = None
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🐾 Siamese GC Capsule Network - Animal Similarity Analyzer", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Create main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel for controls
        left_panel = tk.Frame(main_frame, bg='#ecf0f1', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel for images and results
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # Model loading section
        model_frame = tk.LabelFrame(parent, text="📁 Model Loading", font=('Arial', 12, 'bold'), 
                                   bg='#ecf0f1', fg='#2c3e50', padx=10, pady=10)
        model_frame.pack(fill='x', padx=10, pady=10)
        
        self.model_status = tk.Label(model_frame, text="❌ No model loaded", 
                                    font=('Arial', 10), bg='#ecf0f1', fg='#e74c3c')
        self.model_status.pack(pady=5)
        
        load_model_btn = tk.Button(model_frame, text="Load Model", command=self.load_model,
                                  font=('Arial', 10, 'bold'), bg='#3498db', fg='white',
                                  relief='flat', padx=20, pady=5)
        load_model_btn.pack(pady=5)
        
        # Image loading section
        image_frame = tk.LabelFrame(parent, text="🖼️ Image Loading", font=('Arial', 12, 'bold'), 
                                   bg='#ecf0f1', fg='#2c3e50', padx=10, pady=10)
        image_frame.pack(fill='x', padx=10, pady=10)
        
        load_img1_btn = tk.Button(image_frame, text="Load Image 1", command=lambda: self.load_image(1),
                                 font=('Arial', 10, 'bold'), bg='#27ae60', fg='white',
                                 relief='flat', padx=20, pady=5)
        load_img1_btn.pack(pady=5, fill='x')
        
        load_img2_btn = tk.Button(image_frame, text="Load Image 2", command=lambda: self.load_image(2),
                                 font=('Arial', 10, 'bold'), bg='#27ae60', fg='white',
                                 relief='flat', padx=20, pady=5)
        load_img2_btn.pack(pady=5, fill='x')
        
        # Analysis section
        analysis_frame = tk.LabelFrame(parent, text="🔍 Analysis", font=('Arial', 12, 'bold'), 
                                      bg='#ecf0f1', fg='#2c3e50', padx=10, pady=10)
        analysis_frame.pack(fill='x', padx=10, pady=10)
        
        # Threshold setting
        threshold_frame = tk.Frame(analysis_frame, bg='#ecf0f1')
        threshold_frame.pack(fill='x', pady=5)
        
        tk.Label(threshold_frame, text="Similarity Threshold:", font=('Arial', 10), 
                bg='#ecf0f1').pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(threshold_frame, from_=0.0, to=1.0, resolution=0.01,
                                  orient='horizontal', variable=self.threshold_var,
                                  bg='#ecf0f1', font=('Arial', 9))
        threshold_scale.pack(fill='x', pady=2)
        
        # Analyze button
        self.analyze_btn = tk.Button(analysis_frame, text="🔍 Analyze Similarity", 
                                    command=self.analyze_similarity,
                                    font=('Arial', 12, 'bold'), bg='#e67e22', fg='white',
                                    relief='flat', padx=20, pady=10, state='disabled')
        self.analyze_btn.pack(pady=10, fill='x')
        
        # Results section
        results_frame = tk.LabelFrame(parent, text="📊 Quick Results", font=('Arial', 12, 'bold'), 
                                     bg='#ecf0f1', fg='#2c3e50', padx=10, pady=10)
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.result_label = tk.Label(results_frame, text="No analysis performed", 
                                    font=('Arial', 11, 'bold'), bg='#ecf0f1', fg='#7f8c8d')
        self.result_label.pack(pady=5)
        
        self.similarity_label = tk.Label(results_frame, text="Similarity: N/A", 
                                        font=('Arial', 10), bg='#ecf0f1', fg='#34495e')
        self.similarity_label.pack(pady=2)
        
        self.confidence_label = tk.Label(results_frame, text="Confidence: N/A", 
                                        font=('Arial', 10), bg='#ecf0f1', fg='#34495e')
        self.confidence_label.pack(pady=2)
        
    def setup_right_panel(self, parent):
        """Setup the right panel for images and detailed results"""
        # Images display section
        images_frame = tk.LabelFrame(parent, text="🖼️ Images Comparison", font=('Arial', 12, 'bold'), 
                                    bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        images_frame.pack(fill='x', padx=10, pady=10)
        
        # Image display frame with better layout
        img_display_frame = tk.Frame(images_frame, bg='#f0f0f0')
        img_display_frame.pack(fill='x', pady=10)
        
        # Image 1 with improved layout
        img1_frame = tk.Frame(img_display_frame, bg='#f0f0f0')
        img1_frame.pack(side='left', expand=True, fill='both', padx=10)
        
        tk.Label(img1_frame, text="Image 1", font=('Arial', 12, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(pady=5)
        
        # Container for image with fixed background
        img1_container = tk.Frame(img1_frame, bg='#ecf0f1', relief='sunken', bd=2)
        img1_container.pack(pady=5)
        
        self.img1_label = tk.Label(img1_container, text="No image loaded\n\nClick 'Load Image 1'\nto select an image", 
                                  bg='#bdc3c7', fg='#7f8c8d', width=25, height=15,
                                  font=('Arial', 10), justify='center')
        self.img1_label.pack(padx=5, pady=5)
        
        # Image 2 with improved layout
        img2_frame = tk.Frame(img_display_frame, bg='#f0f0f0')
        img2_frame.pack(side='right', expand=True, fill='both', padx=10)
        
        tk.Label(img2_frame, text="Image 2", font=('Arial', 12, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(pady=5)
        
        # Container for image with fixed background
        img2_container = tk.Frame(img2_frame, bg='#ecf0f1', relief='sunken', bd=2)
        img2_container.pack(pady=5)
        
        self.img2_label = tk.Label(img2_container, text="No image loaded\n\nClick 'Load Image 2'\nto select an image", 
                                  bg='#bdc3c7', fg='#7f8c8d', width=25, height=15,
                                  font=('Arial', 10), justify='center')
        self.img2_label.pack(padx=5, pady=5)
        
        # Add image info display
        info_frame = tk.Frame(images_frame, bg='#f0f0f0')
        info_frame.pack(fill='x', pady=5)
        
        tk.Label(info_frame, text="💡 Tip: Images are displayed larger for visibility but processed at 50×50 for the model", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='#7f8c8d').pack()
        
        # Detailed results section
        details_frame = tk.LabelFrame(parent, text="📋 Detailed Analysis", font=('Arial', 12, 'bold'), 
                                     bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollable text area for detailed results
        text_frame = tk.Frame(details_frame, bg='#f0f0f0')
        text_frame.pack(fill='both', expand=True, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.details_text = tk.Text(text_frame, wrap='word', yscrollcommand=scrollbar.set,
                                   font=('Courier', 10), bg='#ffffff', fg='#2c3e50',
                                   relief='flat', padx=10, pady=10)
        self.details_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.details_text.yview)
        
        # Initialize with welcome message
        welcome_msg = """🐾 Welcome to Siamese GC Capsule Network Analyzer!

📋 Instructions:
1. Load your trained model (.pth file)
2. Load two images you want to compare
3. Adjust the similarity threshold if needed
4. Click 'Analyze Similarity' to get results

🔍 Features:
• Detailed similarity analysis
• Confidence scoring
• Visual comparison with enlarged display
• Embedding analysis
• Interpretation guidance

📏 Image Processing:
• Display: Enlarged for better visibility
• Processing: 50×50 pixels for the model
• Formats: JPG, PNG, BMP, TIFF supported

Ready to analyze animal similarities! 🚀"""
        
        self.details_text.insert('1.0', welcome_msg)
        self.details_text.config(state='disabled')
        
    def load_model(self):
        """Load the trained Siamese model"""
        file_path = filedialog.askopenfilename(
            title="Select Siamese Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load checkpoint with weights_only=False to handle full checkpoint
                try:
                    checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions
                    checkpoint = torch.load(file_path, map_location=self.device)
                
                # Initialize model
                self.model = ImprovedSiameseGCCapsuleNet().to(self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    # Full training checkpoint
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    best_acc = checkpoint.get('best_val_acc', 'N/A')
                    epoch = checkpoint.get('epoch', 'N/A')
                elif isinstance(checkpoint, dict) and 'conv1.weight' in checkpoint:
                    # Direct state dict
                    self.model.load_state_dict(checkpoint)
                    best_acc = 'N/A'
                    epoch = 'N/A'
                else:
                    # Try to load as state dict directly
                    self.model.load_state_dict(checkpoint)
                    best_acc = 'N/A'
                    epoch = 'N/A'
                
                self.model.eval()
                
                self.model_path = file_path
                self.model_status.config(text=f"✅ Model loaded successfully", fg='#27ae60')
                
                # Update threshold if available in checkpoint
                if isinstance(checkpoint, dict) and 'threshold' in checkpoint:
                    self.threshold_var.set(checkpoint['threshold'])
                
                # Update details
                model_info = f"""🤖 MODEL LOADED SUCCESSFULLY!
═══════════════════════════════════════════

📁 File: {os.path.basename(file_path)}
📊 Best Validation Accuracy: {best_acc}
🔄 Training Epoch: {epoch}
💻 Device: {self.device}
🧠 Parameters: {sum(p.numel() for p in self.model.parameters()):,}

Model is ready for similarity analysis! 🚀"""
                
                self.update_details(model_info)
                self.check_ready_state()
                
            except Exception as e:
                error_msg = str(e)
                if "weights_only" in error_msg:
                    detailed_error = """Model Loading Error: PyTorch Security Update

This error occurs due to PyTorch's new security features. 
The model file contains training state that requires special handling.

Possible solutions:
1. Use a model saved with just the state_dict
2. The GUI has been updated to handle this automatically
3. If the error persists, the model file may be corrupted

Technical details: """ + error_msg
                else:
                    detailed_error = f"Failed to load model: {error_msg}"
                
                messagebox.showerror("Model Loading Error", detailed_error)
                self.model_status.config(text="❌ Failed to load model", fg='#e74c3c')
                
                # Update details with error info
                error_info = f"""❌ MODEL LOADING FAILED
═══════════════════════════════════════════

📁 File: {os.path.basename(file_path) if file_path else 'Unknown'}
❌ Error: {error_msg}

🔧 Troubleshooting:
1. Ensure the model file is not corrupted
2. Check that it's a valid PyTorch model file
3. Verify the model architecture matches
4. Try loading a different model file

The model file should be generated from the training notebook
and should contain the trained Siamese GC Capsule Network."""
                
                self.update_details(error_info)
                
    def load_image(self, image_num):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title=f"Select Image {image_num}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path).convert('RGB')
                
                # Create a larger display version (keeping aspect ratio)
                display_size = (250, 250)  # Increased from 200x200
                display_image = image.copy()
                display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # For very small images, scale them up to be more visible
                if display_image.size[0] < 150 or display_image.size[1] < 150:
                    # Calculate scale factor to make image at least 150px on smallest side
                    scale_factor = max(150 / display_image.size[0], 150 / display_image.size[1])
                    new_size = (int(display_image.size[0] * scale_factor), 
                               int(display_image.size[1] * scale_factor))
                    display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Add a border to make images more prominent
                from PIL import ImageOps
                display_image = ImageOps.expand(display_image, border=2, fill='#34495e')
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_image)
                
                if image_num == 1:
                    self.image1_path = file_path
                    self.img1_label.config(image=photo, text="")
                    self.img1_label.image = photo  # Keep a reference
                    
                    # Update label size to accommodate larger images
                    self.img1_label.config(width=display_image.size[0] + 10, 
                                          height=display_image.size[1] + 10)
                else:
                    self.image2_path = file_path
                    self.img2_label.config(image=photo, text="")
                    self.img2_label.image = photo  # Keep a reference
                    
                    # Update label size to accommodate larger images
                    self.img2_label.config(width=display_image.size[0] + 10, 
                                          height=display_image.size[1] + 10)
                
                self.check_ready_state()
                
                # Update status in details
                img_info = f"""📷 IMAGE {image_num} LOADED
═══════════════════════════════════════════

📁 File: {os.path.basename(file_path)}
📏 Original Size: {image.size[0]}x{image.size[1]} pixels
🖼️ Display Size: {display_image.size[0]}x{display_image.size[1]} pixels
⚙️ Processing Size: 50x50 pixels (for model)

The image is displayed larger for better visibility,
but will be processed at 50x50 for the neural network."""
                
                self.update_details(img_info)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                
    def check_ready_state(self):
        """Check if all components are ready for analysis"""
        if self.model and self.image1_path and self.image2_path:
            self.analyze_btn.config(state='normal')
        else:
            self.analyze_btn.config(state='disabled')
            
    def analyze_similarity(self):
        """Perform similarity analysis"""
        if not all([self.model, self.image1_path, self.image2_path]):
            messagebox.showwarning("Warning", "Please load model and both images first!")
            return
            
        try:
            # Update threshold
            self.threshold = self.threshold_var.get()
            
            # Load and preprocess images
            img1 = Image.open(self.image1_path).convert('RGB')
            img2 = Image.open(self.image2_path).convert('RGB')
            
            img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                embed1, embed2 = self.model(img1_tensor, img2_tensor)
                similarity = improved_cosine_similarity(embed1, embed2, temperature=1.5).item()
                
                # Calculate additional metrics
                raw_cosine = F.cosine_similarity(embed1, embed2, dim=1).item()
                euclidean_dist = torch.dist(embed1, embed2, p=2).item()
                
            # Determine result
            is_similar = similarity > self.threshold
            confidence = abs(similarity - self.threshold) / max(self.threshold, 1 - self.threshold)
            
            # Update quick results
            result_text = "✅ SIMILAR" if is_similar else "❌ DIFFERENT"
            result_color = '#27ae60' if is_similar else '#e74c3c'
            
            self.result_label.config(text=result_text, fg=result_color)
            self.similarity_label.config(text=f"Similarity: {similarity:.4f}")
            self.confidence_label.config(text=f"Confidence: {confidence:.4f}")
            
            # Generate detailed analysis
            self.generate_detailed_analysis(similarity, raw_cosine, euclidean_dist, 
                                          is_similar, confidence, embed1, embed2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            
    def generate_detailed_analysis(self, similarity, raw_cosine, euclidean_dist, 
                                 is_similar, confidence, embed1, embed2):
        """Generate detailed analysis report"""
        
        # Get embedding statistics
        embed1_np = embed1.cpu().numpy().flatten()
        embed2_np = embed2.cpu().numpy().flatten()
        
        embed1_mean = np.mean(embed1_np)
        embed1_std = np.std(embed1_np)
        embed2_mean = np.mean(embed2_np)
        embed2_std = np.std(embed2_np)
        
        # Interpretation
        if similarity > 0.9:
            interpretation = "VERY HIGH SIMILARITY - Images are very likely the same breed/type"
        elif similarity > 0.7:
            interpretation = "HIGH SIMILARITY - Images show strong resemblance"
        elif similarity > 0.5:
            interpretation = "MODERATE SIMILARITY - Images have some common features"
        elif similarity > 0.3:
            interpretation = "LOW SIMILARITY - Images show limited resemblance"
        else:
            interpretation = "VERY LOW SIMILARITY - Images are quite different"
            
        # Confidence interpretation
        if confidence > 0.8:
            conf_text = "VERY HIGH - Result is highly reliable"
        elif confidence > 0.6:
            conf_text = "HIGH - Result is reliable"
        elif confidence > 0.4:
            conf_text = "MODERATE - Result has reasonable confidence"
        elif confidence > 0.2:
            conf_text = "LOW - Result should be interpreted carefully"
        else:
            conf_text = "VERY LOW - Result is near threshold, consider manual inspection"
            
        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        detailed_report = f"""🔍 DETAILED SIMILARITY ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════
📅 Analysis Time: {timestamp}
📁 Image 1: {os.path.basename(self.image1_path)}
📁 Image 2: {os.path.basename(self.image2_path)}

🎯 MAIN RESULTS:
═══════════════════════════════════════════════════════════════════════════════
🔸 Final Similarity Score: {similarity:.6f}
🔸 Threshold Used: {self.threshold:.3f}
🔸 Decision: {'SIMILAR' if is_similar else 'DIFFERENT'}
🔸 Confidence Level: {confidence:.4f} ({conf_text})

📊 TECHNICAL METRICS:
═══════════════════════════════════════════════════════════════════════════════
🔸 Raw Cosine Similarity: {raw_cosine:.6f}
🔸 Euclidean Distance: {euclidean_dist:.6f}
🔸 Temperature Scaling: 1.5 (applied)
🔸 Embedding Dimension: 128

🧠 EMBEDDING ANALYSIS:
═══════════════════════════════════════════════════════════════════════════════
📈 Image 1 Embedding Stats:
   • Mean: {embed1_mean:.6f}
   • Std Dev: {embed1_std:.6f}
   • L2 Norm: {torch.norm(embed1).item():.6f}

📈 Image 2 Embedding Stats:
   • Mean: {embed2_mean:.6f}
   • Std Dev: {embed2_std:.6f}
   • L2 Norm: {torch.norm(embed2).item():.6f}

🎯 INTERPRETATION:
═══════════════════════════════════════════════════════════════════════════════
{interpretation}

🔍 DETAILED EXPLANATION:
═══════════════════════════════════════════════════════════════════════════════
The Siamese GC Capsule Network analyzed both images through:

1. 🖼️  FEATURE EXTRACTION:
   • Images resized to 50x50 pixels
   • Processed through convolutional layers
   • Capsule-inspired feature encoding
   • 128-dimensional embedding generation

2. 🔄 SIMILARITY CALCULATION:
   • Cosine similarity between embeddings
   • Temperature scaling (τ=1.5) for better gradients
   • Sigmoid activation for [0,1] range
   • Final score: {similarity:.6f}

3. 🎯 DECISION MAKING:
   • Threshold comparison: {similarity:.6f} {'>' if is_similar else '≤'} {self.threshold:.3f}
   • Confidence based on distance from threshold
   • Result: {'Images are considered SIMILAR' if is_similar else 'Images are considered DIFFERENT'}

4. 📊 RELIABILITY ASSESSMENT:
   • Confidence score: {confidence:.4f}
   • Reliability: {conf_text}
   • Recommendation: {'High confidence in result' if confidence > 0.6 else 'Consider additional validation'}

🔧 MODEL INFORMATION:
═══════════════════════════════════════════════════════════════════════════════
• Architecture: Siamese GC Capsule Network
• Parameters: {sum(p.numel() for p in self.model.parameters()):,}
• Input Size: 50x50x3 (RGB)
• Embedding Size: 128 dimensions
• Training: Extended 25-epoch training with comprehensive analytics

═══════════════════════════════════════════════════════════════════════════════
Analysis completed successfully! 🎉"""
        
        self.update_details(detailed_report)
        
    def update_details(self, text):
        """Update the details text area"""
        self.details_text.config(state='normal')
        self.details_text.delete('1.0', 'end')
        self.details_text.insert('1.0', text)
        self.details_text.config(state='disabled')
        self.details_text.see('1.0')  # Scroll to top


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SiameseGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main() 