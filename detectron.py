import os
import random
import cv2
import json
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# Setup logger (disable TensorBoard if causing issues)
setup_logger(output="./output", distributed_rank=0, name="detectron2")

# Adjust category IDs (fix category id warning)
def adjust_category_ids(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    # Adjust category IDs to start from 1
    for category in data["categories"]:
        category["id"] += 1

    for annotation in data["annotations"]:
        annotation["category_id"] += 1

    # Save the updated file
    with open(annotation_file, "w") as f:
        json.dump(data, f)

# Update dataset annotations
adjust_category_ids("dataset/train/_annotations.coco.json")
adjust_category_ids("dataset/valid/_annotations.coco.json")

# Register the Dataset
dataset_path = "dataset"
register_coco_instances("wall_train", {}, f"{dataset_path}/train/_annotations.coco.json", f"{dataset_path}/train")
register_coco_instances("wall_valid", {}, f"{dataset_path}/valid/_annotations.coco.json", f"{dataset_path}/valid")

# Visualize the Dataset (Optional)
metadata = MetadataCatalog.get("wall_train")
dataset_dicts = DatasetCatalog.get("wall_train")

# Save visualized samples
output_path = "./visualized_samples"
os.makedirs(output_path, exist_ok=True)
for idx, d in enumerate(random.sample(dataset_dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    vis = visualizer.draw_dataset_dict(d)
    output_file = os.path.join(output_path, f"sample_{idx}.jpg")
    cv2.imwrite(output_file, vis.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_file}")

# Configure the Model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wall_train",)
cfg.DATASETS.TEST = ("wall_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000  # Number of iterations
cfg.SOLVER.STEPS = []  # No learning rate steps (fix warning)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Adjust based on your dataset
cfg.OUTPUT_DIR = "./output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the Model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the Model
evaluator = COCOEvaluator("wall_valid", cfg, False, output_dir="./output/")
val_loader = detectron2.data.build_detection_test_loader(cfg, "wall_valid")
print("Running evaluation...")
metrics = detectron2.engine.DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])
print("Evaluation metrics:", metrics)
