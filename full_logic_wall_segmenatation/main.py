import torch
import numpy as np
from PIL import Image
from drawer import Drawer
from model_loading import Mask2FormerInference, DetectronInference, filter_wall_segments


def run_mask2former(image_path: str, model_name: str):
    """
    Runs Mask2Former inference on the input image.

    Parameters:
        image_path (str): Path to the input image.
        model_name (str): Name or path of the Mask2Former model.

    Returns:
        Tuple[torch.Tensor, List[Dict], Dict[int, str]]:
            - segmentation: Segmentation map.
            - segments_info: Metadata for segments.
            - id2label: Label ID to name mapping.
            - mask2former: Mask2FormerInference object.
    """
    print("Running Mask2Former")
    try:
        mask2former = Mask2FormerInference(model_name=model_name, image_path=image_path)
        segmentation, segments_info, id2label = mask2former.post_proces_results()
        return segmentation, segments_info, id2label, mask2former
    except Exception as e:
        print(f"Error during Mask2Former inference: {e}")
        raise


def filter_and_draw(
    mask2former: Mask2FormerInference,
    segmentation: torch.Tensor,
    segments_info: list,
    id2label: dict,
):
    """
    Filters wall segments and visualizes panoptic segmentation.

    Parameters:
        mask2former (Mask2FormerInference): Mask2Former inference object.
        segmentation (torch.Tensor): Segmentation map.
        segments_info (List[Dict]): Metadata for segments.
        id2label (Dict[int, str]): Label ID to name mapping.

    Returns:
        List[np.ndarray]: Extracted wall segment images.
        torch.Tensor: Wall segmentation mask.
    """
    print("Filtering wall segments")
    wall_segments, wall_segmentation = filter_wall_segments(segments_info, segmentation, id2label)

    print("Drawing panoptic segmentation")
    drawer = Drawer(
        image=mask2former.image,
        segmentation=segmentation,
        label_dict=id2label,
        segments_info=segments_info,
    )
    drawer.draw_panoptic_segmentation()

    print("Extracting wall images")
    wall_images = drawer.get_wall_segment_images(wall_segments)
    return wall_images, wall_segmentation


def run_detectron(image_path: str, wall_images: list, wall_segmentation: torch.Tensor):
    """
    Runs Detectron2 inference on wall images.

    Parameters:
        image_path (str): Path to the input image.
        wall_images (List[np.ndarray]): List of wall segment images.
        wall_segmentation (torch.Tensor): Segmentation mask for walls.
    """
    print("Loading Detectron2")
    try:
        detectron = DetectronInference(image_path=image_path, wall_images=wall_images)
        print("Running Detectron2 inference")
        detectron.run_detectron()
    except Exception as e:
        print(f"Error during Detectron2 inference: {e}")
        raise


def main(image_path: str, model_name: str):
    """
    Main function for running the complete segmentation and detection pipeline.

    Parameters:
        image_path (str): Path to the input image.
        model_name (str): Name or path of the Mask2Former model.
    """
    try:
        # Run Mask2Former
        segmentation, segments_info, id2label, mask2former = run_mask2former(image_path, model_name)

        # Filter wall segments and draw results
        wall_images, wall_segmentation = filter_and_draw(
            mask2former, segmentation, segments_info, id2label
        )

        # Run Detectron2
        run_detectron(image_path, wall_images, wall_segmentation)

    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main(
        image_path="../Bathroom/B3.jpg",
        model_name="facebook/mask2former-swin-base-coco-panoptic",
    )
