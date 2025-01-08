from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


# Define label names for your dataset
LABEL_NAMES =  {0: 'Wall', 
                1: 'Painted Wall', 
                2: 'Tail', 
                3: 'Wallpaper', 
                4: 'Wood'}

def load_predictor():
    """
    Configures and loads the Detectron2 model for instance segmentation on CPU.
    """
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "model_final.pth"  # Path to the trained model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABEL_NAMES)  # Number of classes in the dataset
    cfg.MODEL.DEVICE = "cpu"  # Set device to CPU
    return DefaultPredictor(cfg)

import random

def run_detectron(original_image, wall_images, wall_masks=None):
    """
    Runs Detectron2 on individual wall segment images, visualizes predictions with random colors and labels.

    Parameters:
        original_image (numpy.ndarray): Original input image in BGR format.
        wall_images (list): List of wall segment images (numpy arrays in BGR format).
        wall_masks (list, optional): List of wall masks corresponding to each wall image.
                                     If None, no additional mask is applied.
    """
    predictor = load_predictor()

    # Create a copy of the original image for overlay
    visualization_image = original_image.copy()

    for idx, wall_image in enumerate(wall_images):
        # Get the corresponding wall mask if provided
        wall_mask = wall_masks[idx] if wall_masks is not None else None

        # Predictions
        outputs = predictor(wall_image)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()  # Binary masks for each instance
        labels = instances.pred_classes.numpy()  # Class indices for each instance

        # Overlay each mask onto the original image
        for mask, label in zip(masks, labels):
            # Combine with wall mask if provided
            if wall_mask is not None:
                mask = mask & wall_mask

            # Generate a random color
            random_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)

            # Apply the mask with a random color
            visualization_image[mask] = (
                0.7 * visualization_image[mask] + 0.3 * random_color
            ).astype(np.uint8)

            # Add label text to the centroid of the mask
            coords = np.column_stack(np.where(mask))
            if coords.size > 0:
                centroid = coords.mean(axis=0).astype(int)
                cv2.putText(
                    visualization_image,
                    LABEL_NAMES.get(label, f"Class {label}"),
                    (centroid[1], centroid[0]),  # (x, y)
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),  # White text
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )


    # Display the original image with overlayed masks and labels
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title("Detected Masks with Random Colors and Labels")
    plt.axis("off")
    plt.show()



def loading_model(model_name:str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

    return model, processor


def filter_wall_segments(segments_info, segmentation, id2label):
    """
    Filters segments containing 'wall' in their labels.
    """
    wall_segments = []
    wall_segmentation = torch.zeros_like(segmentation)  # Empty mask for wall segments

    for segment in segments_info:
        segment_label_id = segment["label_id"]
        segment_label = id2label[segment_label_id]

        # Check if "wall" is in the label
        if "wall" in segment_label.lower():
            wall_segments.append(segment)
            wall_segmentation[segmentation == segment["id"]] = segment["id"]  # Mask for wall segment

    return wall_segments, wall_segmentation