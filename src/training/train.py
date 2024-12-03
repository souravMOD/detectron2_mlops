import os
import torch
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import (
    AugInput,
    RandomFlip,
    ResizeShortestEdge,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
)
from detectron2.structures import Instances, Boxes

from src.utils.logger import setup_logger
from src.utils.registry import register_datasets

logger = setup_logger(__name__)


# Custom RandomRotation transform to specify the 'indexing' argument
from detectron2.data.transforms import Transform, TransformGen

class CustomRandomRotation(TransformGen):
    def __init__(self, angle_range):
        """
        Args:
            angle_range (tuple): (min_angle, max_angle) in degrees.
        """
        super().__init__()
        self.min_angle, self.max_angle = angle_range

    def get_transform(self, img):
        angle = np.random.uniform(self.min_angle, self.max_angle)
        h, w = img.shape[:2]
        return CustomRotationTransform(angle, image_size=(h, w))

class CustomRotationTransform(Transform):
    def __init__(self, angle, image_size):
        super().__init__()
        self.angle = angle
        self._image_size = image_size  # Store image size for coordinate transformations

    def apply_image(self, img):
        import cv2
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated_img = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated_img

    def apply_coords(self, coords):
        """
        Apply rotation to coordinates.

        Args:
            coords (ndarray): An array of shape (M, 2), where M is the number of points.

        Returns:
            ndarray: Rotated coordinates of the same shape.
        """
        angle_rad = np.deg2rad(self.angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle],
        ])
        # Calculate the center of the image
        h, w = self._image_size
        center = np.array([w / 2, h / 2])
        # Shift coordinates to origin
        coords_shifted = coords - center
        # Apply rotation
        coords_rotated = coords_shifted @ rotation_matrix.T
        # Shift coordinates back
        coords_rotated += center
        return coords_rotated

    def apply_box(self, boxes):
        """
        Apply the rotation transform to bounding boxes.

        Args:
            boxes (ndarray): An array of shape (N, 4), where N is the number of boxes.

        Returns:
            ndarray: The transformed boxes of shape (N, 4).
        """
        # Extract corners for all boxes
        x0 = boxes[:, 0]
        y0 = boxes[:, 1]
        x1 = boxes[:, 2]
        y1 = boxes[:, 3]
        
        # Create arrays of corners for all boxes
        corners = np.stack([
            np.stack([x0, y0], axis=1),
            np.stack([x1, y0], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x0, y1], axis=1)
        ], axis=1)  # Shape: (N, 4, 2)
        
        # Reshape corners to (N * 4, 2) for batch processing
        corners_flat = corners.reshape(-1, 2)
        
        # Apply rotation to all corners
        rotated_corners_flat = self.apply_coords(corners_flat)
        
        # Reshape back to (N, 4, 2)
        rotated_corners = rotated_corners_flat.reshape(-1, 4, 2)
        
        # Compute new bounding boxes
        x_coords = rotated_corners[:, :, 0]
        y_coords = rotated_corners[:, :, 1]
        x0_new = x_coords.min(axis=1)
        y0_new = y_coords.min(axis=1)
        x1_new = x_coords.max(axis=1)
        y1_new = y_coords.max(axis=1)
        
        transformed_boxes = np.stack([x0_new, y0_new, x1_new, y1_new], axis=1)
        return transformed_boxes


def custom_mapper(dataset_dict):
    """
    Custom mapper function for applying data augmentations to training data.

    Args:
        dataset_dict (dict): A dictionary containing image and annotation information.

    Returns:
        dict: The updated dataset dictionary with augmented image and annotations.
    """
    # Make a deep copy to avoid modifying the original dataset_dict
    dataset_dict = dataset_dict.copy()

    # Read the image from the file path
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Define augmentations
    augmentations = [
        RandomFlip(horizontal=True, vertical=False),
        ResizeShortestEdge(
            short_edge_length=(640, 800),
            max_size=1333,
            sample_style="choice"
        ),
        CustomRandomRotation(angle_range=(-10, 10)),  # Use custom rotation
        RandomBrightness(0.8, 1.2),
        RandomContrast(0.8, 1.2),
        RandomSaturation(0.8, 1.2),
    ]

    # Apply augmentations
    aug_input = AugInput(image)
    transforms = aug_input.apply_augmentations(augmentations)

    # Convert the augmented image to a tensor
    image_tensor = torch.as_tensor(
        aug_input.image.transpose(2, 0, 1).astype("float32")
    )

    # Update the dataset dictionary with the augmented image
    dataset_dict["image"] = image_tensor

    # Apply transforms to annotations, if available
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_size=aug_input.image.shape[:2]
            )
            for obj in dataset_dict.pop("annotations")
        ]

        # Create Instances object
        instances = Instances(aug_input.image.shape[:2])

        # Convert list of bounding boxes into a single NumPy array
        bbox_list = [anno["bbox"] for anno in annos]
        bbox_array = np.array(bbox_list, dtype=np.float32)  # Ensure correct data type

        # Convert the NumPy array into a PyTorch tensor
        bbox_tensor = torch.from_numpy(bbox_array)

        # Assign to gt_boxes
        instances.gt_boxes = Boxes(bbox_tensor)

        # Assign ground truth classes
        instances.gt_classes = torch.tensor(
            [anno["category_id"] for anno in annos], dtype=torch.int64
        )

        dataset_dict["instances"] = instances

    return dataset_dict


class AugmentedTrainer(DefaultTrainer):
    """
    Custom Trainer that uses a custom data mapper for augmentations.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build the training data loader with a custom mapper.

        Args:
            cfg (CfgNode): The configuration node.

        Returns:
            DataLoader: A PyTorch DataLoader with custom mapping.
        """
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def train(cfg_path: str):
    """
    Train the model using the provided configuration file.

    Args:
        cfg_path (str): Path to the configuration file.
    """
    from src.utils.config_manager import load_config

    # Load configurations
    config = load_config(cfg_path)
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']

    # Register datasets
    splits = {
        "train": dataset_config['train_split'],
        "val": dataset_config['val_split'],
        "test": dataset_config['test_split'],
    }
    register_datasets(
        dataset_config['json_file'],
        dataset_config['base_dir'],
        splits
    )
    logger.info("Datasets registered successfully.")

    # Set up Detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(model_config['config_file'])
    )
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.DATALOADER.NUM_WORKERS = training_config['num_workers']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        model_config['weights']
    )
    cfg.SOLVER.IMS_PER_BATCH = model_config['ims_per_batch']
    cfg.SOLVER.BASE_LR = training_config['base_lr']
    cfg.SOLVER.MAX_ITER = training_config['max_iter']
    cfg.SOLVER.STEPS = training_config['steps']
    cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = model_config['batch_size_per_image']
    cfg.MODEL.RETINANET.NUM_CLASSES = model_config['num_classes']
    cfg.OUTPUT_DIR = training_config['output_dir']
    cfg.MODEL.DEVICE = 'cuda'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory created at {cfg.OUTPUT_DIR}")

    # Start training with the AugmentedTrainer
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
