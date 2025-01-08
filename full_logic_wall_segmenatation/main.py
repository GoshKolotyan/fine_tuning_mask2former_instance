import cv2
import torch
import numpy as np
from PIL import Image
from model_loading import loading_model, filter_wall_segments,  run_detectron
from drawer import draw_panoptic_segmentation, get_wall_segment_images  # Import the Detectron2 integration function
import matplotlib.pyplot as plt


def main(image_path):
    # Load model and processor
    model, processor = loading_model("facebook/mask2former-swin-base-coco-panoptic")

    # Load image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=0.5
    )[0]

    # Extract segmentation and segments_info
    segmentation = results["segmentation"].cpu()
    segments_info = results["segments_info"]

    # Filter wall segments
    wall_segments, wall_segmentation = filter_wall_segments(
        segments_info, segmentation, model.config.id2label
    )
    # Draw panoptic segmentation
    # draw_panoptic_segmentation(
    #     segmentation=segmentation,
    #     model=model,
    #     segments_info=segments_info,
    # )

    # Visualize individual walls
    wall_images = get_wall_segment_images(image, wall_segments, segmentation)
    # Visualize individual wall images for debugging
    # wall_images = get_wall_segment_images(image, wall_segments, segmentation)

    # Create subplots to display all wall images
    num_walls = len(wall_images)

    # for idx, wall_image in enumerate(wall_images):
    #     plt.imshow(cv2.cvtColor(wall_image, cv2.COLOR_BGR2RGB))  # Ensure correct color order
    #     plt.title(f"Wall Segment {idx + 1}")
    #     plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    # Convert PIL image to OpenCV format (BGR)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # print("Running detectron")

    # # Run Detectron2 on the image, optionally using the wall mask
    run_detectron(original_image=image_cv2,wall_images=wall_images)  # Optional: Add a wall mask if needed


# Run the main function
if __name__ == "__main__":
    main("./Bathroom/B4.jpg")
