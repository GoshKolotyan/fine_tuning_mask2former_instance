import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Define label names for your dataset
LABEL_NAMES =  {0: 'Wall', 
                1: 'Painted Wall', 
                2: 'Tail', 
                3: 'Wallpaper', 
                4: 'Wood'}

# Streamlit App Title
st.title("Detectron2 Instance Segmentation with Label Names")

# Load the trained model configuration
@st.cache_resource  # Cache the predictor to avoid reloading on every interaction
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join("experiments/experiment_20250105_024026", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABEL_NAMES)  # Adjust based on your dataset
    return DefaultPredictor(cfg)

predictor = load_predictor()

# Upload images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error(f"Could not read image: {uploaded_file.name}")
            continue

        # Perform predictions
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Extract predictions
        masks = instances.pred_masks.numpy()
        labels = instances.pred_classes.numpy()

        # Create a subplot to display results
        fig, axes = plt.subplots(1, len(labels) + 1, figsize=(5 * (len(labels) + 1), 5))
        fig.suptitle(f"Predictions for {uploaded_file.name}")

        # Original image
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Plot each predicted instance
        for i, (mask, label) in enumerate(zip(masks, labels)):
            overlay = img.copy()
            overlay[mask] = [0, 255, 0]  # Green mask overlay
            blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

            label_name = LABEL_NAMES.get(label, f"Class {label}")
            axes[i + 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f"Label: {label_name}")
            axes[i + 1].axis("off")

        # Display the plot
        st.pyplot(fig)

        # Optionally save the result
        save_dir = "predictions_streamlit"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"pred_{uploaded_file.name}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        st.success(f"Saved prediction visualization to {output_path}")
