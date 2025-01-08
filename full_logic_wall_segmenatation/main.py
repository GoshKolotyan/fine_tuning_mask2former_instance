import cv2
import torch
import numpy as np
from PIL import Image
from model_loading import loading_model, filter_wall_segments, run_detectron, post_proces_results
from drawer import (
    draw_panoptic_segmentation,
    get_wall_segment_images,
)  # Import the Detectron2 integration function
import matplotlib.pyplot as plt


def main(image_path:str,
         model_name:str)->None:
    print("Loading Model")
    model, processor = loading_model(model_name)

    print("Runing post-processing")
    print("Image-Path",image_path,10*"--")
    segmentation, segments_info, id2label = post_proces_results(
        model=model,
        processor=processor,
        image_path=image_path,
    )

    # Filter wall segments
    wall_segments, wall_segmentation = filter_wall_segments(
        segments_info, segmentation, id2label
    )
    # Draw panoptic segmentation
    draw_panoptic_segmentation(
        segmentation=segmentation,
        model=model,
        segments_info=segments_info,
    )

     # Visualize individual wall images for debugging
    # wall_images = get_wall_segment_images(image, wall_segments, segmentation)


    # Convert PIL image to OpenCV format (BGR)
    # image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # print("Running detectron")

    # # Run Detectron2 on the image, optionally using the wall mask
    # run_detectron(
    #     original_image=image_cv2, wall_images=wall_images
    # )  # Optional: Add a wall mask if needed


# Run the main function
if __name__ == "__main__":
    main(image_path="../Bathroom/B4.jpg",
         model_name="facebook/mask2former-swin-base-coco-panoptic")
