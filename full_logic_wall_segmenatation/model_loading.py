import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hyperparams import LABEL_NAMES
from typing import List, Dict, Tuple
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


def load_predictor():
    """
    Configures and loads the Detectron2 model for instance segmentation on CPU.

    Raises:
        FileNotFoundError: If configuration or weights file is not found.
        ValueError: If LABEL_NAMES is not defined or empty.

    Returns:
        DefaultPredictor: Configured Detectron2 predictor.
    """
    try:
        # Validate LABEL_NAMES
        if not isinstance(LABEL_NAMES, dict) or not LABEL_NAMES:
            raise ValueError("LABEL_NAMES must be a non-empty dictionary.")

        # Initialize configuration
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "model_final.pth"  # Path to the trained model weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABEL_NAMES)  # Number of classes in the dataset
        cfg.MODEL.DEVICE = "cpu"  # Set device to CPU

        return DefaultPredictor(cfg)
    except FileNotFoundError as e:
        raise FileNotFoundError("Configuration or weights file not found. Ensure all required files are in place.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the predictor: {e}") from e

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



def loading_model(model_name:str)->Tuple[any]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

    return model, processor

def post_proces_results(
    model: Mask2FormerForUniversalSegmentation,
    processor: AutoImageProcessor,
    image_path: str,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, List[Dict], Dict[int, str]]:
    """
    Performs inference and post-processes panoptic segmentation results using a given model and processor.

    Parameters:
        model (Mask2FormerForUniversalSegmentation): Pre-trained segmentation model.
        processor (AutoImageProcessor): Processor for image preprocessing and postprocessing.
        image_path (str): Path to the input image.
        threshold (float, optional): Threshold for post-processing. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, List[Dict], Dict[int, str]]: 
            - segmentation: Segmentation map as a torch.Tensor.
            - segments_info: Metadata for each segment.
            - id2label: Label ID to name mapping.

    Raises:
        ValueError: If the image file is not found, corrupted, or if invalid parameters are provided.
    """
    try:        
        # Load and verify image
        image = Image.open(image_path)
        image.verify()  # Ensure the image is not corrupted
        image = Image.open(image_path)  # Reload the image after verification
    except FileNotFoundError:
        raise ValueError(f"Image not found at {image_path}. Please provide a valid path.")
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Model inference
    with torch.inference_mode():  # Use inference_mode for faster inference
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=threshold
    )[0]

    segmentation = results["segmentation"].cpu()
    segments_info = results["segments_info"]

    # Retrieve label mapping from the model configuration
    id2label = model.config.id2label

    return segmentation, segments_info, id2label



def filter_wall_segments(
    segments_info: List[Dict],
    segmentation: torch.Tensor,
    id2label: Dict[int, str]
) -> Tuple[List[Dict], torch.Tensor]:
    """
    Filters segments containing 'wall' in their labels.

    Parameters:
        segments_info (List[Dict]): List of dictionaries containing segment metadata.
        segmentation (torch.Tensor): Segmentation map with segment IDs.
        id2label (Dict[int, str]): Mapping from label IDs to label names.

    Returns:
        Tuple[List[Dict], torch.Tensor]: 
            - A list of segment dictionaries for 'wall' segments.
            - A segmentation mask (torch.Tensor) with 'wall' segments highlighted.

    Raises:
        ValueError: If `segments_info`, `segmentation`, or `id2label` are invalid or empty.
    """
    # Validate inputs
    if not isinstance(segments_info, list) or not all(isinstance(seg, dict) for seg in segments_info):
        raise ValueError("segments_info must be a list of dictionaries.")
    if not isinstance(segmentation, torch.Tensor):
        raise ValueError("segmentation must be a torch.Tensor.")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError("id2label must be a non-empty dictionary.")

    # Initialize outputs
    wall_segments = []
    wall_segmentation = torch.zeros_like(segmentation)  # Empty mask for wall segments

    # Filter segments containing 'wall'
    for segment in segments_info:
        segment_label_id = segment.get("label_id")
        if segment_label_id is None or segment_label_id not in id2label:
            continue  # Skip if label_id is missing or invalid

        segment_label = id2label[segment_label_id]

        # Check if "wall" is in the label
        if "wall" in segment_label.lower():
            wall_segments.append(segment)
            wall_segmentation[segmentation == segment["id"]] = segment["id"]  # Mask for wall segment

    return wall_segments, wall_segmentation
