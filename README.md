# 🐛 Fall Armyworm Supervised AI Detection  
Capstone Project – AI/ML Bootcamp

### Team Members:
- Oluwapelumi Babalola

---

## 📌 Project Overview
The Fall Armyworm (Spodoptera frugiperda) is a destructive agricultural pest responsible for severe crop losses, particularly in maize production. Early detection is vital for preventing outbreaks and reducing yield losses.

This project applies **Supervised Machine Learning** and **Computer Vision** to detect Fall Armyworm from images. The final trained model will be exported to **ONNX format** for easy deployment in real-world environments.

---

## ✅ Objectives
- Build a supervised AI model for FAW detection/classification.
- Train and evaluate using a custom dataset + augmentations.
- Optimize the model for small size and deployment speed.
- Export the final best-performing model in ONNX format.
- Document a reproducible end-to-end ML pipeline.

---

## 🧪 Project Tasks & Workflow
1. **Data Loading**
   - Custom FAW dataset + public dataset sources
   - Image annotations and labeling

2. **Data Preprocessing**
   - Image resizing, normalization, and augmentation
   - Data split: Train / Validation / Test

3. **Model Development**
   - Model choice: CNN or Object Detection model
   - Loss function, optimizer, hyperparameters
   - Training & tuning

4. **Evaluation Metrics**
   - Accuracy / Precision / Recall / F1 Score (Classification)
   - mAP (Object Detection)
   - Model size and inference speed

5. **Model Export**
   - Convert model to ONNX format: `model.onnx`

---

## 📂 Repository Structure
```bash
FAW-Detection/
│
├── data/
│   └── dataset_links.md
│
├── notebooks/
│   ├── 1_FAW_Detection_Project_.ipynb
│
├── models/
│   └── faw_model(2).onnx
│
└── README.md
