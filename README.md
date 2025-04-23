# 🌿 AI Crop Disease Detector

A Deep Learning-powered web application built with **TensorFlow** and **Streamlit** to detect plant diseases from leaf images. This tool is designed to help farmers and agriculturalists quickly identify diseases and take appropriate actions.

## 📸 Demo

Upload a clear image of a plant leaf, and the model will predict the most likely disease from a dataset of 38 different classes.

## 🚀 Features

- 📷 Upload leaf images via a simple web interface
- 🧠 AI-based classification trained on PlantVillage dataset
- 🌐 Google search link for more disease info
- 💡 User-friendly, responsive UI
- 📦 Easily deployable with Streamlit

## 🧠 Model Details

- **Architecture**: CNN with multiple convolutional + dense layers
- **Framework**: TensorFlow / Keras
- **Trained On**: 38-class Kaggle PlantVillage dataset
- **Input Size**: 128x128 RGB images
- **Accuracy**: ~90% validation accuracy

## 🛠️ How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/KARTIK2005/AI-Crop-Disease-Detector.git
   cd AI-Crop-Disease-Detector

2. Run the app using Streamlit:
   
   streamlit run app.py

3. Upload a plant leaf image and get results!

🗂️ Project Structure

AI-Crop-Disease-Detector/
├── app.py                  # Streamlit web app
<br>
├── crop_disease_model.h5   # Trained Keras model
<br>
├── class_labels.json       # Label mapping
<br>
├── history.pkl             # (Optional) Training history
<br>
├── README.md               # Project overview
<br>
└── requirements.txt        # Dependencies

📦 Requirements

Python 3.7+
Streamlit
TensorFlow
Pillow
NumPy

You can install all dependencies using:

pip install -r requirements.txt


