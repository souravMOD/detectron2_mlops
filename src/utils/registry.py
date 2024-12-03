import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

def register_datasets(json_file, base_dir, splits):
    """
    Register COCO datasets for training, validation, and testing.

    Args:
        json_file (str): Path to the COCO JSON file.
        base_dir (str): Path to the base directory containing the dataset.
        splits (dict): Dictionary with split names as keys and relative paths as values.
                       Example: {"train": "train/images", "val": "val/images", "test": "test/images"}
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    categories = [cat["name"] for cat in coco_data.get("categories", [])]

    for split, relative_path in splits.items():
        images_dir = os.path.join(base_dir, relative_path)
        annotations_file = os.path.join(base_dir, f"{split}\{split}.json")

        DatasetCatalog.register(f"{split}_dataset", lambda x=annotations_file, y=images_dir: load_coco_json(x, y))
        MetadataCatalog.get(f"{split}_dataset").set(thing_classes=categories)

        print(f"Registered {split}_dataset with {annotations_file}")
