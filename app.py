import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model
model = load_model("crop_disease_model.h5")

# Load class labels
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Sidebar
st.sidebar.title("🌿 AI Crop Disease Detector")
st.sidebar.markdown("Built with **TensorFlow** and **Streamlit**\n\nUpload a plant leaf image to identify diseases instantly.")
st.sidebar.markdown("### 🌱 How to use:")
st.sidebar.markdown("1. Upload an image of the plant leaf.\n"
                    "2. Our model will analyze the leaf and provide the predicted disease.\n"
                    "3. You can click on the disease name to get more information on Google.")

# Main title
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("### 📷 Upload an image of the plant leaf")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns to display the image and result side by side
    col1, col2 = st.columns([2, 3])  # Adjust column ratio as needed

    # Display the image in the first column
    with col1:
        st.image(img, caption='📸 Uploaded Leaf Image', use_container_width=True)

    # Preprocess the image for prediction
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the disease
    with col2:
        with st.spinner("🔍 Analyzing..."):
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_label = index_to_class[predicted_index]

        # Stylish Result Box with rounded corners and shadow effect
        st.markdown(f"""
        <style>
            .result-box {{
                padding: 20px;
                background-color: #006400;
                color: white;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                font-size: 18px;
                text-align: center;
            }}
            .result-box:hover {{
                background-color: #228B22;
                cursor: pointer;
            }}
        </style>
        <div class="result-box">
            ✅ Predicted Disease: <b>{predicted_label}</b>
        </div>
        """, unsafe_allow_html=True)

        # Add Google search link for the disease name with styled hover effect
        disease_url = f"https://www.google.com/search?q={predicted_label.replace(' ', '+')}"
        st.markdown(f"🔍 [Click here for more information on {predicted_label}]( {disease_url} )", unsafe_allow_html=True)

        st.markdown("💡 *Note: For best results, use a clear, close-up image of the leaf.*")

else:
    st.info("Please upload a plant leaf image to proceed.")

# Footer with modern style and centered text
st.markdown("---")
st.markdown("<div style='text-align: center; color: #2E8B57; font-size: 14px;'>Made with ❤️ to help farmers</div>", unsafe_allow_html=True)


