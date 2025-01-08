from typing import List, Dict
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as mpatches


class Drawer:
    """
    A utility class for visualizing panoptic segmentation results and extracting individual wall segment images.

    Attributes:
        image (np.ndarray): Original input image.
        segmentation (torch.Tensor): Segmentation map with segment IDs.
        label_dict (Dict[int, str]): Dictionary mapping label IDs to label names.
        segments_info (List[Dict]): List of dictionaries with segment metadata.
    """

    def __init__(
        self,
        image: Image.Image,
        segmentation: torch.Tensor,
        label_dict: Dict[int, str],
        segments_info: List[Dict],
    ):
        """
        Initializes the Drawer class.

        Parameters:
            image (Image.Image): Original input image.
            segmentation (torch.Tensor): Segmentation map with segment IDs.
            label_dict (Dict[int, str]): Dictionary mapping label IDs to label names.
            segments_info (List[Dict]): List of dictionaries with segment metadata.
        """
        self.image = np.array(image) if isinstance(image, Image.Image) else image
        self.segmentation = segmentation
        self.label_dict = label_dict
        self.segments_info = segments_info

    def draw_panoptic_segmentation(self):
        """
        Visualizes the panoptic segmentation results, including a legend for each segment.
        """
        viridis = cm.get_cmap("viridis", int(torch.max(self.segmentation).item()) + 1)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.segmentation, cmap=viridis)
        instances_counter = defaultdict(int)
        handles = []

        for segment in self.segments_info:
            segment_id = segment["id"]
            segment_label_id = segment["label_id"]
            segment_label = self.label_dict.get(segment_label_id, f"Unknown-{segment_label_id}")
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id / torch.max(self.segmentation).item())
            handles.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1))
        ax.axis("off")
        plt.title("Panoptic Segmentation Visualization")
        plt.show()

    def get_wall_segment_images(self, wall_segments: List[Dict]) -> List[np.ndarray]:
        """
        Extracts individual wall segment images with all other parts blacked out.

        Parameters:
            wall_segments (List[Dict]): List of wall segment dictionaries.

        Returns:
            List[np.ndarray]: List of masked images for each wall segment.
        """
        wall_images = []

        for segment in wall_segments:
            mask = self.segmentation == segment["id"]
            black_image = np.zeros_like(self.image)
            masked_image = black_image.copy()
            masked_image[mask] = self.image[mask]
            wall_images.append(masked_image)

        return wall_images
