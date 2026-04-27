# 🧠 AI-Based Brain Tumor Detection System

## 📌 Project Title
**AI-Based Brain Tumor Detection, Classification and Segmentation using EfficientNet-B0, V-Net, Grad-CAM, and Medical Chatbot**

---

## 🌟 Overview
This project presents an AI-powered brain tumor analysis system that performs **detection, classification, and segmentation** of MRI brain scans. It integrates deep learning models with medical assistance tools to improve diagnosis and decision-making.

The system combines:
- **EfficientNet-B0** for classification  
- **Grad-CAM** for visual interpretability  
- **V-Net** for tumor segmentation  
- **NLP-based Chatbot** for medical assistance  

A **Streamlit-based web application** enables users to upload MRI images and receive real-time diagnostic insights.

---

## 🎯 Objectives
- Early and accurate tumor detection  
- Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)  
- Tumor region segmentation and visualization  
- Improve interpretability using Grad-CAM  
- Provide AI-based medical assistance via chatbot  

---

## 🧠 Techniques Used

### 📊 Data Processing
- Image Resizing  
- Normalization (0–1 scaling)  
- Data Augmentation  

---

### 🤖 Model Architecture
- **EfficientNet-B0** → Classification  
- **Grad-CAM** → Heatmap visualization  
- **V-Net** → Tumor segmentation  
- **Transfer Learning**  

---

### 🔍 Classification Classes
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

---

## 📂 Dataset
Dataset used: **BraTS (Brain Tumor Segmentation Dataset)**  

⚠️ Dataset is not uploaded due to size limitations.

👉 Download from:  
https://www.kaggle.com/

---

## ⚙️ Model Details
```json
{
  "model": "EfficientNet-B0 + V-Net + Grad-CAM",
  "epochs": 50,
  "batch_size": 32,
  "optimizer": "Adam",
  "accuracy": "High (Experimental Results)"
}

 🌐 Web Application (Streamlit)
🔧 Installation
pip install -r requirements.txt
streamlit run app.py

🖥️ Features
🧠 Brain Tumor Analyzer
Upload MRI image
Detect tumor presence
Classify tumor type
Highlight affected region

🔥 Visualization
Grad-CAM heatmaps
Region highlighting for interpretability

🧩 Segmentation
V-Net based tumor boundary detection
Region extraction for severity analysis

💬 Medical Chatbot
Answers queries about tumors
Suggests treatments
Provides hospital recommendations


📁 Project Structure
brain-tumor-analysis/
│── app.py
│── requirements.txt
│── model/
│── src/
│── images/
│── README.md

📊 Results
High classification accuracy
Accurate tumor localization
Improved interpretability using Grad-CAM
Reliable segmentation with V-Net

🚀 Future Work
Deploy on cloud
Improve segmentation accuracy
Add real-time clinical integration

👩‍💻 Authors
S ABHIGNYA
D SHUBHANG

📜 License

This project is for academic and research purposes only.
