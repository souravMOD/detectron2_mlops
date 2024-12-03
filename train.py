import os
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from sklearn.model_selection import train_test_split


def register_coco_split(dataset_name, json_file, image_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Registers train, val, and test splits dynamically for a single COCO dataset.

    Args:
        dataset_name (str): Base name of the dataset (e.g., 'my_dataset').
        json_file (str): Path to the COCO JSON file.
        image_dir (str): Path to the image directory.
        split_ratios (tuple): Ratios for train, val, and test splits (default is 70% train, 20% val, 10% test).
    """
    # Load the full dataset
    dataset_dicts = load_coco_json(json_file, image_dir)
    
    # Split the dataset
    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    train_dicts, temp_dicts = train_test_split(dataset_dicts, test_size=(1 - train_ratio), random_state=42)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_dicts, test_dicts = train_test_split(temp_dicts, test_size=(1 - val_ratio_adjusted), random_state=42)
    
    # Register the splits
    DatasetCatalog.register(f"{dataset_name}_train", lambda d=train_dicts: d)
    MetadataCatalog.get(f"{dataset_name}_train").set(json_file=json_file, image_root=image_dir)

    DatasetCatalog.register(f"{dataset_name}_val", lambda d=val_dicts: d)
    MetadataCatalog.get(f"{dataset_name}_val").set(json_file=json_file, image_root=image_dir)

    DatasetCatalog.register(f"{dataset_name}_test", lambda d=test_dicts: d)
    MetadataCatalog.get(f"{dataset_name}_test").set(json_file=json_file, image_root=image_dir)

    print(f"Registered splits: {dataset_name}_train, {dataset_name}_val, {dataset_name}_test")


# Example usage
json_file = r"D:\DATASET\84fd3c3f-dd3e-400b-ac5a-d18ca712c3d0\weedcoco.json"
image_dir = r"D:\DATASET\84fd3c3f-dd3e-400b-ac5a-d18ca712c3d0\images"
register_coco_split("my_dataset", json_file, image_dir)

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    
    # Use the dynamically registered datasets
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.RETINANET.NUM_CLASSES = 22  # Ensure this matches your dataset

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
