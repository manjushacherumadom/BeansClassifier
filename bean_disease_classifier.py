#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit tensorflow Pillow


# In[1]:


import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

# Load the pre-trained model (make sure the .keras model file is available)
model = load_model('BeansClassify.keras')

# Class labels (replace these with your actual class names)
class_names = ['Angular_leaf_spot', 'bean_rust', 'healthy']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to fit MobileNet input size
    img = np.array(img)  # Convert to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Streamlit application interface
st.title('Bean Disease Classifier')
st.write('Upload an image of a bean to identify if it is healthy or diseased.')

# File upload widget
uploaded_file = st.file_uploader("Choose a bean image...", type=["jpg", "png", "jpeg"])

# If a file is uploaded
if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Bean Image.", use_column_width=True)

        # Preprocess the image
        img = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence_score = np.max(prediction) * 100  # Confidence score as percentage

        # Display the prediction and confidence score
        st.write(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {confidence_score:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.write("Waiting for an image...")


# #go to cmd prompt and run the command - streamlit run bean_disease_classifier.py --server.port 8502

# In[ ]:




