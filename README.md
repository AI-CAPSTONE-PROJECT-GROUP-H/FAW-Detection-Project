# Fall Armyworm Detection using YOLOv8

## 📌 Project Overview
The Fall Armyworm (Spodoptera frugiperda) is a destructive agricultural pest responsible for severe crop losses, particularly in maize production. Early detection is vital for preventing outbreaks and reducing yield losses.

This project applies **Supervised Machine Learning** and **Computer Vision** to detect Fall Armyworm from images using **YOLOv8 object detection**. The final trained model is exported to **ONNX format** for deployment in real-world agricultural monitoring systems.

---

## 🎯 Objectives
- ✅ Build a YOLOv8-based object detection model for FAW detection
- ✅ Train and evaluate using a custom Roboflow dataset
- ✅ Optimize model for deployment (YOLOv8n - Nano variant)
- ✅ Export to ONNX format for cross-platform compatibility
- ✅ Document reproducible ML pipeline in Google Colab

---

## 📊 Dataset

### Source
- **Primary Dataset**: Roboflow Fall Armyworm Dataset
- **Format**: YOLOv8 (images + YOLO txt annotations)
- **Classes**: [fall-armyworm-egg, fall-armyworm-frass, fall-armyworm-larva, fall-armyworm-larval-damage, healthy-maize, maize-streak-disease]

### Dataset Split
| Split | Images | Purpose |
|-------|--------|---------|
| Train | 7317 | Model training |
| Valid | 1246 | Hyperparameter tuning |
| Test | 1271 | Final evaluation |

### Data Augmentation
- Horizontal flip (50% probability)
- Minimal augmentation for faster training

---

## 🏗️ Model Architecture

**Model**: YOLOv8 Nano (yolov8n.pt)
- **Input Size**: 416×416 pixels
- **Backbone**: CSPDarknet
- **Neck**: PAN (Path Aggregation Network)
- **Head**: Decoupled detection head
- **Parameters**: ~3.2M
- **Model Size**: ~6 MB (ONNX)

### Why YOLOv8 Nano?
1. ⚡ Fast inference (~30-50 FPS on CPU)
2. 📦 Small model size for deployment
3. 🎯 Good balance of speed vs accuracy
4. 🔧 Easy ONNX export

---

## 🔧 Implementation

### Environment
- **Platform**: Google Colab (T4 GPU)
- **Framework**: Ultralytics YOLOv8
- **Language**: Python 3.12
- **Key Libraries**: 
  - `ultralytics` (YOLOv8)
  - `opencv-python` (image processing)
  - `onnxruntime` (inference)

### Training Configuration
```python
epochs = 25
batch_size = 32
image_size = 416×416
optimizer = SGD
device = GPU (T4)
patience = 5 (early stopping)
```

### Training Time
- **With T4 GPU**: ~15-20 minutes
- **With CPU**: ~3-4 hours ❌

---

## 📈 Results

### Model Performance
| Metric | Value |
|--------|-------|
| **mAP50** | 0.6255 |
| **mAP50-95** | 0.5577 |
| **Precision** | 0.4914 |
| **Recall** | 0.6746 |

---

## 🚀 Usage

### 1. Training
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=25,
    imgsz=416,
    batch=32,
    device=0
)
```

### 2. Inference (PyTorch)
```python
from ultralytics import YOLO

model = YOLO('faw_model.pt')
results = model.predict('test_image.jpg', conf=0.25)
```

### 3. Inference (ONNX)
```python
import onnxruntime as ort

session = ort.InferenceSession('faw_model.onnx')
# [Preprocessing code]
outputs = session.run(None, inputs)
```

---

## 📁 Project Structure
```
fall-armyworm-detection/
│
├── notebooks/
│   └── FAW_Detection_Project_.ipynb     # Main training notebook
|
├── data/
│   └── dataset_links.md
│
├── models/
│   ├── faw_model.pt             # PyTorch weights
│   └── faw_model.onnx           # ONNX export
│
└── README.md
```

---

## 🎓 Lessons Learned

### Challenges
1. **GPU Configuration**: Initially trained on CPU (3+ hours) before switching to T4 GPU (20 mins)
2. **ONNX Post-processing**: YOLOv8 output shape required custom parsing
3. **Low Detection Rate**: Early models had 0 detections due to:
   - Incorrect post-processing for output shape `[1, 10, 3549]`
   - Threshold tuning needed

### Solutions
1. ✅ Enabled T4 GPU in Colab (`device=0`)
2. ✅ Custom post-processing for YOLOv8 ONNX format
3. ✅ Lower confidence threshold testing (0.25 → 0.15 → 0.05)

---

## 🔮 Future Improvements

1. **More Training**: Increase epochs to 50-100 for better accuracy
2. **Data Augmentation**: Add more augmentation techniques
3. **Ensemble Models**: Combine multiple models for robustness
4. **Edge Deployment**: Test on mobile/embedded devices
5. **Video Processing**: Real-time detection in video streams

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Roboflow Dataset](https://universe.roboflow.com/)
- [Fall Armyworm Information](https://www.fao.org/fall-armyworm/)

---

## 👥 Team Members:

- Wilson Divine Wopara
- Oluwapelumi Babalola
- Samuel Egbon
- Glory Godwin
- Tamukong Elvis Achu
- Zaharaddeen Abdulsalam
- Ibrahim David Mohammed
- AJAYI ADEOLA ABRAHAM
- Azeez Abdulhakeem
- Mohammed Muye Umar
- Egharevba Eghosa
- Khadijat hassan-dogo
- Taiwo Odewabi
- Olorunfemi Emmanuel Damilare
- Adetona Michael by name
---
