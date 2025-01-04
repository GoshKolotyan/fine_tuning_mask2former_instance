import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances

# Register the test dataset (if not already registered)
# Replace the paths with your test dataset paths
test_dataset_name = "wall_test"
test_annotations_path = "dataset/test/_annotations.coco.json"
test_images_path = "dataset/test"
register_coco_instances(test_dataset_name, {}, test_annotations_path, test_images_path)

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for predictions
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Adjust based on your dataset
cfg.DATASETS.TEST = (test_dataset_name,)

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Load the test dataset
test_metadata = MetadataCatalog.get(test_dataset_name)
test_dataset_dicts = DatasetCatalog.get(test_dataset_name)

# Directory to save predictions
output_dir = "./predictions"
os.makedirs(output_dir, exist_ok=True)

# Perform inference and save visualizations
for d in test_dataset_dicts:
    # Load image
    img = cv2.imread(d["file_name"])
    
    # Make predictions
    outputs = predictor(img)
    
    # Visualize predictions
    visualizer = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the visualized prediction
    output_file = os.path.join(output_dir, os.path.basename(d["file_name"]))
    cv2.imwrite(output_file, vis.get_image()[:, :, ::-1])
    print(f"Saved prediction visualization to {output_file}")
