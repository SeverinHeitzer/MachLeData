"""Streamlit dashboard for model demos and monitoring.

The dashboard lets stakeholders inspect sample predictions, model
quality metrics, and lightweight operational health signals.
"""

import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from machledata.data import load_sample_paths
from machledata.infer import predict_image
from machledata.metrics import compute_class_distribution, compute_detection_statistics
from machledata.model import build_model_config


def draw_detections_on_image(image_path: Path, detections):
    """Draw bounding boxes and labels on image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Draw label
        label_text = f"{detection.class_name} {detection.confidence:.2f}"
        draw.text((x1, y1 - 10), label_text, fill="red")

    return image


def main() -> None:
    """Render the dashboard for local development and demos."""
    st.set_page_config(page_title="MachLeData Detection", layout="wide")
    st.title("🎯 MachLeData Object Detection Dashboard")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio(
            "Select Mode",
            ["📸 Live Detection", "📊 Statistics", "🎬 Batch Processing", "ℹ️ About"],
        )

        st.subheader("Model Configuration")
        model_name = st.selectbox(
            "Model",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
            help="YOLO model variant",
        )
        image_size = st.slider(
            "Image Size",
            min_value=320,
            max_value=1280,
            value=640,
            step=64,
            help="Input image size for model",
        )
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum detection confidence",
        )
        model_path = st.text_input(
            "Custom Model Path (optional)",
            help="Path to saved model file",
        )

    config = build_model_config(
        model_name=model_name,
        image_size=image_size,
        confidence_threshold=confidence,
    )

    # Mode: Live Detection
    if mode == "📸 Live Detection":
        st.header("Live Object Detection")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp"],
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Save temporarily and run detection
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.getbuffer())

                with st.spinner("Running detection..."):
                    detections = predict_image(
                        temp_path,
                        model_path=model_path or None,
                        config=config,
                    )

                # Display results
                if detections:
                    st.success(f"Found {len(detections)} object(s)")

                    # Draw and display
                    annotated_image = draw_detections_on_image(temp_path, detections)
                    st.image(annotated_image, caption="Detections", use_column_width=True)

                    # Show detections table
                    detection_data = [
                        {
                            "Class": d.label,
                            "Confidence": f"{d.confidence:.4f}",
                            "X1": f"{d.bbox_xyxy[0]:.0f}",
                            "Y1": f"{d.bbox_xyxy[1]:.0f}",
                            "X2": f"{d.bbox_xyxy[2]:.0f}",
                            "Y2": f"{d.bbox_xyxy[3]:.0f}",
                        }
                        for d in detections
                    ]
                    st.dataframe(detection_data, use_container_width=True)
                else:
                    st.info("No objects detected in this image")

        with col2:
            st.subheader("Model Info")
            st.json(
                {
                    "Model": model_name,
                    "Image Size": image_size,
                    "Confidence": confidence,
                }
            )

    # Mode: Statistics
    elif mode == "📊 Statistics":
        st.header("Detection Statistics")

        sample_paths = load_sample_paths("data/samples")

        if sample_paths:
            st.subheader(f"Analyzing {len(sample_paths)} Sample Images")

            with st.spinner("Running detection on samples..."):
                all_detections = []
                for image_path in sample_paths:
                    detections = predict_image(
                        image_path,
                        model_path=model_path or None,
                        config=config,
                    )
                    all_detections.append(detections)

            # Compute statistics
            stats = compute_detection_statistics(all_detections)
            class_dist = compute_class_distribution(all_detections)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", int(stats["total_images"]))
            with col2:
                st.metric("Total Detections", int(stats["total_detections"]))
            with col3:
                st.metric(
                    "Avg per Image",
                    f"{stats['average_detections_per_image']:.2f}",
                )
            with col4:
                st.metric(
                    "Avg Confidence",
                    f"{stats['average_confidence']:.3f}",
                )

            if class_dist:
                st.subheader("Class Distribution")
                st.bar_chart(class_dist)

            st.json(stats)
        else:
            st.info("No sample images found in data/samples")

    # Mode: Batch Processing
    elif mode == "🎬 Batch Processing":
        st.header("Batch Image Processing")

        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Process All Images"):
                temp_dir = Path("/tmp/batch_processing")
                temp_dir.mkdir(exist_ok=True)

                results = {}
                progress_bar = st.progress(0)

                for idx, file in enumerate(uploaded_files):
                    # Save file
                    temp_path = temp_dir / file.name
                    temp_path.write_bytes(file.getbuffer())

                    # Run detection
                    detections = predict_image(
                        temp_path,
                        model_path=model_path or None,
                        config=config,
                    )

                    results[file.name] = [
                        {
                            "class": d.label,
                            "confidence": float(d.confidence),
                            "bbox": list(d.bbox_xyxy),
                        }
                        for d in detections
                    ]

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                st.success("Batch processing complete!")

                # Download results
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="Download Results (JSON)",
                    data=results_json,
                    file_name="detections.json",
                    mime="application/json",
                )

                # Display summary
                total_detections = sum(len(dets) for dets in results.values())
                st.metric("Total Detections", total_detections)

    # Mode: About
    else:  # About
        st.header("About MachLeData")
        st.markdown(
            """
            **MachLeData** is a YOLO-based object detection pipeline with MLOps practices.

            ### Features
            - 📸 Real-time object detection using YOLO v8
            - 📊 Statistics and analytics dashboard
            - 🎬 Batch processing capabilities
            - 🔧 Configurable model and inference parameters
            - 🚀 Production-ready API

            ### Supported Models
            - YOLOv8 Nano (yolov8n) - Fastest, least accurate
            - YOLOv8 Small (yolov8s)
            - YOLOv8 Medium (yolov8m)
            - YOLOv8 Large (yolov8l) - Slowest, most accurate

            ### How to Use
            1. Select mode from the sidebar
            2. Configure model parameters
            3. Upload images or process samples
            4. Review detection results and statistics
            """
        )


if __name__ == "__main__":
    main()


