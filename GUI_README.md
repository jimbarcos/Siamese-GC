# 🐾 Siamese GC Capsule Network - GUI Application

A comprehensive graphical user interface for analyzing animal image similarities using your trained Siamese GC Capsule Network model.

## 🚀 Features

- **Model Loading**: Import your trained `.pth` model files
- **Image Comparison**: Load and compare two images side by side
- **Detailed Analysis**: Comprehensive similarity analysis with technical metrics
- **Visual Interface**: User-friendly GUI with real-time results
- **Confidence Scoring**: Reliability assessment of similarity predictions
- **Embedding Analysis**: Deep technical insights into model predictions

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- Windows/macOS/Linux
- At least 4GB RAM
- GPU support optional (CUDA)

### Python Dependencies
Install the required packages:
```bash
pip install -r requirements_gui.txt
```

Or install manually:
```bash
pip install torch torchvision Pillow numpy
```

## 🎯 Usage Instructions

### 1. Launch the Application
```bash
python siamese_gui.py
```

### 2. Load Your Trained Model
1. Click **"Load Model"** button
2. Select your trained model file (e.g., `best_siamese_gc_capsule_model.pth`)
3. Wait for "✅ Model loaded successfully" confirmation

### 3. Load Images for Comparison
1. Click **"Load Image 1"** to select the first image
2. Click **"Load Image 2"** to select the second image
3. Supported formats: JPG, JPEG, PNG, BMP, TIFF

### 4. Adjust Settings (Optional)
- Use the **Similarity Threshold** slider to set your decision boundary
- Default: 0.5 (values above = similar, below = different)

### 5. Analyze Similarity
1. Click **"🔍 Analyze Similarity"** button
2. View quick results in the left panel
3. Read detailed analysis in the right panel

## 📊 Understanding Results

### Quick Results Panel
- **Decision**: SIMILAR ✅ or DIFFERENT ❌
- **Similarity Score**: 0.0 to 1.0 (higher = more similar)
- **Confidence**: How reliable the prediction is

### Detailed Analysis Report
The comprehensive report includes:

#### 🎯 Main Results
- Final similarity score with 6 decimal precision
- Threshold comparison and decision
- Confidence level with interpretation

#### 📊 Technical Metrics
- Raw cosine similarity
- Euclidean distance
- Temperature scaling effects
- Embedding dimensions

#### 🧠 Embedding Analysis
- Statistical analysis of both image embeddings
- Mean, standard deviation, and L2 norms
- Feature space insights

#### 🔍 Detailed Explanation
- Step-by-step processing explanation
- Feature extraction details
- Similarity calculation methodology
- Decision-making process

## 🎨 Interface Overview

### Left Panel (Controls)
- **📁 Model Loading**: Load your trained model
- **🖼️ Image Loading**: Import images for comparison
- **🔍 Analysis**: Threshold setting and analyze button
- **📊 Quick Results**: Immediate results display

### Right Panel (Results)
- **🖼️ Images Comparison**: Side-by-side image display
- **📋 Detailed Analysis**: Comprehensive scrollable report

## 🔧 Technical Details

### Model Architecture
- **Input Size**: 50×50×3 (RGB images)
- **Embedding Size**: 128 dimensions
- **Architecture**: Siamese GC Capsule Network
- **Similarity Function**: Improved cosine similarity with temperature scaling

### Image Processing
- Automatic resizing to 50×50 pixels
- ImageNet normalization applied
- RGB conversion for consistency
- Tensor preprocessing for model input

### Similarity Calculation
1. **Feature Extraction**: Both images → 128D embeddings
2. **Cosine Similarity**: Dot product of normalized embeddings
3. **Temperature Scaling**: τ=1.5 for better gradient flow
4. **Sigmoid Activation**: Maps to [0,1] range
5. **Threshold Comparison**: Decision based on user threshold

## 🎯 Tips for Best Results

### Image Selection
- Use clear, well-lit animal photos
- Avoid heavily cropped or blurry images
- Ensure animals are clearly visible
- Similar lighting conditions work best

### Threshold Setting
- **0.3-0.4**: Very strict similarity (only very similar images)
- **0.5**: Balanced (default recommendation)
- **0.6-0.7**: More lenient (allows moderate similarities)

### Interpreting Confidence
- **High Confidence (>0.6)**: Trust the result
- **Moderate Confidence (0.4-0.6)**: Reasonable reliability
- **Low Confidence (<0.4)**: Consider manual verification

## 🚨 Troubleshooting

### Common Issues

#### "Failed to load model"
- Ensure the model file is a valid `.pth` file
- Check that the file isn't corrupted
- Verify the model was trained with the same architecture

#### "Failed to load image"
- Check image file format is supported
- Ensure image file isn't corrupted
- Try converting to JPG format

#### "Analysis failed"
- Ensure both model and images are loaded
- Check available system memory
- Restart the application if needed

#### GUI doesn't start
- Verify Python version (3.7+)
- Install missing dependencies
- Check tkinter is available (usually included with Python)

### Performance Tips
- **GPU Usage**: Automatically detected and used if available
- **Memory**: Close other applications for better performance
- **Image Size**: Larger images are automatically resized

## 📁 File Structure
```
siamese_gui.py          # Main GUI application
requirements_gui.txt    # Python dependencies
GUI_README.md          # This documentation
best_siamese_gc_capsule_model.pth  # Your trained model (from training)
```

## 🔗 Related Files
- **Training Notebook**: `Siamese_GC_Capsule_Network_Study_CLEAN.ipynb`
- **Model Files**: Generated from training (`.pth` files)
- **Training History**: `training_history.json`

## 🎉 Example Workflow

1. **Start Application**: `python siamese_gui.py`
2. **Load Model**: Select `best_siamese_gc_capsule_model.pth`
3. **Load Images**: Choose two animal photos to compare
4. **Set Threshold**: Adjust to 0.5 (or your preference)
5. **Analyze**: Click analyze and review detailed results
6. **Interpret**: Use confidence scores and detailed explanation

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Ensure your model file is compatible
4. Check Python and dependency versions

---

🐾 **Happy analyzing with your Siamese GC Capsule Network!** 🚀 