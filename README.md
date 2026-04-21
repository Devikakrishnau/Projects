# Projects
🧠 Agentic AI Tumor Detection System

An AI-powered system for brain tumor detection, risk assessment, and automated radiology report generation using MRI images.

📌 Overview

This project uses Deep Learning (ResNet-18) and a multi-agent architecture to:

Detect brain tumors from MRI scans
Classify tumor types
Estimate tumor size
Assess clinical risk level
Generate automated radiology reports
🚀 Features
🧠 Tumor Classification (Glioma, Meningioma, Pituitary, No Tumor)
📏 Tumor Size Estimation (in cm)
⚠️ Clinical Risk Assessment (Low / Medium / High)
🔍 Explainable AI (Grad-CAM visualization) (optional)
📄 Automated PDF Radiology Report
💻 Interactive Web UI using Gradio
🏗️ System Architecture
MRI Image
   ↓
Image Preprocessing
   ↓
ResNet-18 Model
   ↓
Tumor Classification
   ↓
Tumor Size Estimation
   ↓
Risk Assessment
   ↓
Report Generation
🧩 Modules
Module	Description
Image Processing	Preprocess MRI images
Classification Agent	Predict tumor type
Size Agent	Estimate tumor size
Risk Agent	Assess clinical risk
Report Generator	Create PDF reports
Orchestrator	Coordinates all agents
📂 Project Structure
Agentic_Tumor/
│
├── backend/
│   ├── agents/
│   ├── trained_models/
│   ├── utils/
│
├── frontend/
│   ├── app.py
│   ├── report_generator.py
│
├── dataset/
├── requirements.txt
└── README.md
📊 Dataset
Source: Kaggle Brain Tumor MRI Dataset
Classes:
Glioma
Meningioma
Pituitary
No Tumor
Total Images: ~5700
🧠 Model Details
Model: ResNet-18 CNN
Framework: PyTorch
Input Size: 224 × 224
Loss Function: CrossEntropyLoss
Optimizer: Adam
📈 Training Results
Accuracy: ~99%
Loss: Reduced from 99.87 → 5.91
⚙️ Installation
1. Clone the repository
git clone https://github.com/Devikakrishnau/Projects.git
cd agentic-tumor-ai
2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Application
python frontend/app.py

Open in browser:

http://127.0.0.1:7860
🧪 Model Training

To retrain the model:

python backend/train_model.py

Model will be saved at:

backend/trained_models/tumor_model.pt
📄 Report Generation

The system generates a radiology-style PDF report including:

Patient details
Tumor prediction
Tumor size
Risk level
Clinical findings
Impression
⚠️ Limitations
Model trained on limited dataset
Performance depends on MRI orientation
May not generalize to all clinical images
🔮 Future Improvements
Use larger medical datasets (e.g., BraTS)
Improve tumor segmentation accuracy
Deploy as cloud-based diagnostic tool
Integrate hospital systems
📚 References
He K. et al., Deep Residual Learning for Image Recognition, CVPR
Kaggle Brain Tumor MRI Dataset
Medical Imaging Research Papers
👩‍💻 Author

Devika Krishna
