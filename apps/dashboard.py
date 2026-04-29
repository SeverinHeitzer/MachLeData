import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import time

API_URL = "http://localhost:8000/detect"

st.set_page_config(page_title="YOLO Detection Dashboard", layout="wide")
st.title("📸 YOLO Object Detection – Live Monitoring")

# Sidebar for configuration
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
return_annotated = st.sidebar.checkbox("Show annotated image", True)

# Main area: upload or capture
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display original
    original = Image.open(uploaded_file)
    col1.image(original, caption="Original Image", use_column_width=True)
    
    # Send to API
    files = {"file": uploaded_file.getvalue()}
    params = {"return_annotated": return_annotated}
    
    with st.spinner("Detecting..."):
        response = requests.post(API_URL, files=files, params=params)
    
    if response.status_code == 200:
        data = response.json()
        detections = data["detections"]
        
        # Metrics display
        col2.metric("Number of objects detected", len(detections))
        
        if detections:
            df = pd.DataFrame(detections)
            fig = px.bar(df, x="class_name", y="confidence", color="class_name",
                         title="Detection Confidence per Class")
            col2.plotly_chart(fig, use_container_width=True)
            
            # Detection table
            st.subheader("Detailed detections")
            st.dataframe(df[["class_name", "confidence", "bbox"]])
        else:
            st.info("No objects detected with the current confidence threshold.")
        
        # Show annotated image if requested
        if return_annotated and data.get("annotated_image_base64"):
            import base64
            img_bytes = base64.b64decode(data["annotated_image_base64"])
            annotated_img = Image.open(io.BytesIO(img_bytes))
            st.image(annotated_img, caption="Annotated Result", use_column_width=True)
    else:
        st.error(f"API error: {response.status_code}")