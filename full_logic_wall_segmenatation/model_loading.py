import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from hyperparams import LABEL_NAMES
from typing import List, Dict, Tuple, Optional
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class DetectronInference:
    def __init__(self,
                image_path: str, 
                wall_images: List[np.ndarray]):
        """
        Initializes the DetectronInference class.

        Parameters:
            model_name (str): Name of the Detectron2 model.
            original_image (np.ndarray): Original input image in BGR format.
            wall_images (List[np.ndarray]): List of wall segment images in BGR format.
            wall_masks (List[np.ndarray], optional): List of wall masks for each wall segment.
        """
        self.original_image = cv2.cvtColor(cv2.imread(image_path), 
                                           cv2.COLOR_BGR2RGB)

        self.wall_images = wall_images

    def load_predictor(self):
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
            cfg.MODEL.WEIGHTS = "model_final.pth"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABEL_NAMES)
            cfg.MODEL.DEVICE = "cpu"

            return DefaultPredictor(cfg)
        except FileNotFoundError as e:
            raise FileNotFoundError("Configuration or weights file not found.") from e
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the predictor: {e}") from e
    def run_detectron(self):
        """
        Runs Detectron2 on wall segment images, visualizes predictions with 
        random colors and labels.
        """
        predictor = self.load_predictor()

        # Create a copy of the original image for overlay
        visualization_image = self.original_image.copy()

        for idx, wall_image in enumerate(self.wall_images):
            
            # Predictions
            outputs = predictor(wall_image)
            instances = outputs["instances"].to("cpu")
            masks = instances.pred_masks.numpy()  # Binary masks for each instance
            labels = instances.pred_classes.numpy()  # Class indices for each instance

            # Overlay each mask 
            for mask, label in zip(masks, labels):
                random_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
                visualization_image[mask] = (0.7 * visualization_image[mask] + 0.3 * random_color).astype(np.uint8)

                coords = np.column_stack(np.where(mask))
                if coords.size > 0:
                    centroid = coords.mean(axis=0).astype(int)
                    cv2.putText(
                        visualization_image,
                        LABEL_NAMES.get(label, f"Class {label}"),
                        (centroid[1], centroid[0]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Masks with Random Colors and Labels")
        plt.axis("off")
        plt.show()

class Mask2FormerInference:
    """
    Class for running Mask2Former panoptic segmentation inference on a given image.

    Attributes:
        model_name (str): Name or path of the pre-trained model.
        image_path (str): Path to the input image.
        model (Mask2FormerForUniversalSegmentation): Pre-trained Mask2Former model.
        processor (AutoImageProcessor): Processor for image preprocessing and postprocessing.
        image (PIL.Image): Loaded and verified image.
    """

    def __init__(self, model_name: str, image_path: str):
        """
        Initializes the Mask2FormerInference class by loading the model, processor, and input image.

        Parameters:
            model_name (str): Name or path of the pre-trained model.
            image_path (str): Path to the input image.

        Raises:
            ValueError: If the image is missing, corrupted, or if the model cannot be loaded.
        """
        self.model_name = model_name
        self.image_path = image_path
        try:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model or processor for {self.model_name}: {e}")
        try:
            self.image = Image.open(self.image_path)
            self.image.verify()  # Ensure the image is not corrupted
            self.image = Image.open(self.image_path)  # Reload after verification
        except (FileNotFoundError, ValueError):
            raise ValueError(f"Invalid or corrupted image at {self.image_path}. Please provide a valid path.")
    
    def post_proces_results(self, threshold: float = 0.5) -> Tuple[torch.Tensor, List[Dict], Dict[int, str]]:
        """
        Performs inference and post-processes panoptic segmentation results.

        Parameters:
            threshold (float, optional): Threshold for post-processing. Defaults to 0.5.

        Returns:
            Tuple[torch.Tensor, List[Dict], Dict[int, str]]: 
                - segmentation: Segmentation map as a torch.Tensor.
                - segments_info: Metadata for each segment.
                - id2label: Label ID to name mapping.
        """
        inputs = self.processor(images=self.image, 
                                return_tensors="pt")

        # Model inference
        with torch.inference_mode():  # Use inference_mode for faster inference
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[self.image.size[::-1]], 
            threshold=threshold
        )[0]

        segmentation = results["segmentation"].cpu()
        segments_info = results["segments_info"]

        # Retrieve label mapping from the model configuration
        id2label = self.model.config.id2label

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
