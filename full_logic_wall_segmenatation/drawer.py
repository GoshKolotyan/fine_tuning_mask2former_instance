import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import List
from matplotlib import cm
from collections import defaultdict


# Function to visualize panoptic segmentation
def draw_panoptic_segmentation(segmentation: torch.Tensor,
                               model: transformers.models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentation,
                               segments_info: List )-> None:
    # Get the color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation).item() + 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(segmentation, cmap=viridis)
    instances_counter = defaultdict(int)
    handles = []

    # For each segment, create a legend entry
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id / torch.max(segmentation).item())
        handles.append(mpatches.Patch(color=color, label=label))

    # Add the legend
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1))
    ax.axis("off")
    plt.show()

def get_wall_segment_images(image:Image, 
                            wall_segments:List, 
                            segmentation_map:torch.Tensor)->List[torch.Tensor]:
    """
    Extracts individual wall segment images with all other parts blacked out.

    Parameters:
        image (PIL.Image or np.ndarray): Original input image.
        wall_segments (list): List of wall segment dictionaries.
        segmentation_map (torch.Tensor): Segmentation map with segment IDs.

    Returns:
        list: List of images (numpy arrays) for each wall segment.
    """
    # Convert image to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # List to store individual wall images
    wall_images = []

    for segment in wall_segments:
        # Extract individual wall mask
        mask = segmentation_map == segment["id"]

        # Create a black image
        black_image = np.zeros_like(image)

        # Apply mask to the original image
        masked_image = black_image.copy()
        masked_image[mask] = image[mask]

        # Append the masked image to the list
        wall_images.append(masked_image)

    return wall_images
