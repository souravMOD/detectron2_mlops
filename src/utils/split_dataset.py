import os
import json
import shutil
from sklearn.model_selection import train_test_split

def split_coco_json(input_json, output_directory, image_directory, split_ratios=(0.7, 0.2, 0.1)):
    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    random_seed = 42  # Define a random seed for reproducibility

    # Load the input COCO JSON file
    with open(input_json, 'r') as f:
        coco_data = json.load(f)

    # Extract the images and annotations
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    # Map image IDs to their annotations
    image_id_to_annotations = {}
    for ann in annotations:
        image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

    # Perform train-test-validation split
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=random_seed)
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_images, test_images = train_test_split(temp_images, test_size=(1 - val_ratio_adjusted), random_state=random_seed)

    # Helper function to create a COCO dataset
    def create_coco_subset(images_subset):
        annotations_subset = []
        for img in images_subset:
            annotations_subset.extend(image_id_to_annotations.get(img["id"], []))
        return {
            "images": images_subset,
            "annotations": annotations_subset,
            "categories": coco_data["categories"]
        }

    # Create subsets
    train_data = create_coco_subset(train_images)
    val_data = create_coco_subset(val_images)
    test_data = create_coco_subset(test_images)

    # Ensure output directories exist
    os.makedirs(output_directory, exist_ok=True)
    train_dir = os.path.join(output_directory, "train")
    val_dir = os.path.join(output_directory, "val")
    test_dir = os.path.join(output_directory, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save subsets as JSON files
    train_json = os.path.join(output_directory, "train.json")
    val_json = os.path.join(output_directory, "val.json")
    test_json = os.path.join(output_directory, "test.json")

    with open(train_json, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_json, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_json, 'w') as f:
        json.dump(test_data, f, indent=2)

    # Copy images to respective folders
    def copy_images(images_subset, target_dir):
        for img in images_subset:
            src_path = os.path.join(image_directory, img["file_name"])
            dest_path = os.path.join(target_dir, img["file_name"])
            if not os.path.exists(src_path):
                print(f"Warning: Image {src_path} not found!")
                continue
            shutil.copy(src_path, dest_path)

    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)
    copy_images(test_images, test_dir)

    print(f"Train, validation, and test datasets saved to {output_directory}")
    print(f"Images organized into {train_dir}, {val_dir}, and {test_dir}")


# Example Usage
if __name__ == "__main__":
    input_json = r"D:\DATASET\detectron2\datasets\weedcoco.json"  # Path to your COCO JSON file
    output_directory = r"D:\DATASET\detectron2\datasets"         # Directory to save the split files
    image_directory = r"D:\DATASET\detectron2\datasets\images"            # Directory containing all images
    split_ratios = (0.7, 0.2, 0.1)                               # Train, validation, test split ratios

    split_coco_json(input_json, output_directory, image_directory, split_ratios)
