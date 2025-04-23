import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load model
model = load_model('crop_disease_model.h5')

# Load class label mapping
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_indices.items()}

# Load test image
img_path = 'test_leaf.jpg'  # change if your image has a different name
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_label = index_to_class[predicted_index]

# Show result
print("Predicted Disease:", predicted_label)
