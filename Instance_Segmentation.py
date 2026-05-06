"""
Instance_Segmentation.py

This script implements a full deep learning pipeline for image segmentation using PyTorch
and segmentation_models_pytorch. It includes:

- Data loading and preprocessing
- Custom dataset classes
- Model creation (UNet-based)
- Training & validation loops
- Evaluation metrics (IoU, Dice)
- Hyperparameter tuning using Optuna
- Visualization utilities

NOTE:
All file paths have been removed and replaced with placeholders.
Please update them according to your local environment.
"""

import os
import csv
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import shutil
import optuna
import pandas as pd
import albumentations as albu
import ndjson
import re
from torch.utils.data import Dataset
import logging
from segmentation_models_pytorch import utils
import cv2
import random
from torch.utils.tensorboard import SummaryWriter
from skimage.morphology import remove_small_objects
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from collections import Counter


# =========================
# FILE PATHS (UPDATE REQUIRED)
# =========================

# TODO: Replace all paths below with your local file paths

KOP_FILE_PATH = "YOUR_PATH_HERE"          # Path to KOP CSV file
OP_FILE_PATH = "YOUR_PATH_HERE"           # Path to OP CSV file
CSV_FILE_PATH = "YOUR_PATH_HERE"          # Path to additional CSV file
JSON_FILE_PATH = "YOUR_PATH_HERE"         # Path to NDJSON file

TRAIN_KOP_DIR = "YOUR_PATH_HERE"          # Training images directory (KOP)
MASK_KOP_DIR = "YOUR_PATH_HERE"           # Mask directory (KOP)

TRAIN_OP_DIR = "YOUR_PATH_HERE"           # Training images directory (OP)
MASK_OP_DIR = "YOUR_PATH_HERE"            # Mask directory (OP)

TRAIN_GOP_DIR = "YOUR_PATH_HERE"          # Training images directory (GOP)
MASK_GOP_DIR = "YOUR_PATH_HERE"           # Mask directory (GOP)

TRAIN_GKOP_DIR = "YOUR_PATH_HERE"         # Training images directory (GKOP)
MASK_GKOP_DIR = "YOUR_PATH_HERE"          # Mask directory (GKOP)

TEST_FILE_PATH = "YOUR_PATH_HERE"         # Test CSV file
TEST_DIR = "YOUR_PATH_HERE"               # Test images directory
TEST_MASK_DIR = "YOUR_PATH_HERE"          # Test mask directory

new_labels_dir = "YOUR_PATH_HERE"         # New labels directory
mask_folder = "YOUR_PATH_HERE"            # Mask folder
labels_csv = "YOUR_PATH_HERE"             # Labels CSV

# =========================
# Utility Functions
# =========================

def parse_json(string):
    """
    Safely parse a JSON string.

    Args:
        string (str): JSON string

    Returns:
        dict: Parsed JSON or empty dict if NaN
    """
    if pd.isna(string):
        return {}
    return json.loads(string)


def read_csv(file_path):
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        raise



def process_image_ids(df, column_name):
    """Extract and process image IDs from a DataFrame column, removing all file extensions."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    
    image_ids = df[column_name].tolist()
    
    # Remove all file extensions and handle specific cases like .tif.png
    processed_ids = []
    skipped_files = []
    
    for filename in image_ids:
        print(filename)
        if filename.endswith('.tif.png'):
            processed_id = filename[:-8]  # Remove .tif.png (8 characters)
        elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
            processed_id = filename.rsplit('.', 1)[0]  # Remove the last extension
        else:
            #logging.warning(f"Unknown file extension for {filename}")
            processed_id = filename
        
        img_path = find_image_file(processed_id)
        if img_path:
            processed_ids.append(processed_id)
        else:
            skipped_files.append(filename)
    
    logging.info(f'Processed {len(processed_ids)} image IDs')
    logging.info(f'Skipped {len(skipped_files)} image IDs due to missing files: {skipped_files}')
    
    return processed_ids, len(skipped_files)


def process_new_image_ids(df, column_name, root_dir):
    """Extract and process image IDs from a DataFrame column, removing all file extensions."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    
    image_ids = df[column_name].tolist()
    
    # Range of missing image IDs
    missing_ids_range = [f"0°-Ansicht_Probe_{i}" for i in range(1522, 1676)]
    
    # Remove all file extensions and handle specific cases like .tif.png
    processed_ids = []
    skipped_files = []
    
    for filename in image_ids:
        if filename.endswith('.tif.png'):
            processed_id = filename[:-8]  # Remove .tif.png (8 characters)
        elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
            processed_id = filename.rsplit('.', 1)[0]  # Remove the last extension
        else:
            logging.warning(f"Unknown file extension for {filename}")
            processed_id = filename
        
        # Check if the processed ID is in the missing IDs range
        if processed_id in missing_ids_range:
            skipped_files.append(filename)
            continue

        img_path = find_new_image_file(processed_id, root_dir)
        if img_path:
            processed_ids.append(processed_id)
        else:
            skipped_files.append(filename)
    
    logging.info(f'Processed {len(processed_ids)} image IDs')
    #logging.info(f'Skipped {len(skipped_files)} image IDs due to missing files: {skipped_files}')
    
    return processed_ids

def find_new_image_file(image_id, directory):
    """
    Check if the image file exists in the specified directory.
    
    Args:
        image_id (str): The base name of the image file without extension.
        directory (str): The directory to search for the image file.
        
    Returns:
        str: The file path if the image file exists, otherwise None.
    """
    extensions = ['.jpg', '.png', '.tif', '.tif.png']
    
    for ext in extensions:
        file_path = os.path.join(directory, image_id + ext)
        if os.path.isfile(file_path):
            return file_path
    
    return None

def find_image_file(img_id):
    """
    Search for an image file with different extensions.

    NOTE:
    Replace directory path with your dataset path.

    Args:
        img_id (str): Image ID without extension

    Returns:
        str or None: Full path if found
    """
    # TODO: Replace with your dataset directory
    base_dir = "YOUR_IMAGE_DIRECTORY"

    extensions = ['.tif', '.jpg', '.png', '.tif.png']
    for ext in extensions:
        img_path = os.path.join(base_dir, img_id + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def read_ndjson(file_path):
    """Read data from an NDJSON file and return a list of external IDs."""
    try:
        with open(file_path, 'r') as f:
            data = ndjson.load(f)
        image_list = [d['data_row']['external_id'] for d in data]
        logging.info(f'Successfully read data from {file_path}')
        return image_list
    except Exception as e:
        logging.error(f'Error reading data from {file_path}: {e}')
        raise

def clean_image_list(image_list):
    """Clean image file names by removing the ' - Copy' suffix and file extensions."""
    cleaned_list = [re.sub(r'( - Copy)?\.(jpg|png|tif)$', '', filename) for filename in image_list if isinstance(filename, str)]
    logging.info(f'Cleaned {len(cleaned_list)} image file names')
    return cleaned_list

def load_test_data(test_dir, exclude_ids):
    '''Loads the data from the csv. removes .tif extension of the file to get image ids.'''
    test_list = []
    for filename in os.listdir(test_dir):
        base_name = filename.split('.')[0]
        if filename.endswith('.tif') and base_name in exclude_ids:
            test_list.append(filename)
    return test_list

# =========================
# Dataset Class
# =========================

class SegmentationDataset(Dataset):
    """
    Custom dataset for Instance segmentation.

    Handles:
    - Image loading
    - Mask loading
    - Augmentation
    - Preprocessing
    """
    CLASSES = ['Pores', 'Open Pores', 'Weld']
    PIXEL_TO_LABEL = {60: 'Background', 78: 'Open Pores', 185: 'Pores', 216: 'Weld'}

    def __init__(self, images_dir, id_list, output_dir=None, augmentation=None, classes=None, preprocessing=None, verbose=False):
        self.images_dir = images_dir
        self.id_list = self.filter_missing_files(id_list)
        self.output_dir = output_dir
        self.classes = [self.CLASSES.index(cls) for cls in classes] if classes else list(range(len(self.CLASSES)))
        self.class_values = classes if classes else self.CLASSES
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.verbose = verbose  # Control debug prints
        self.class_counts = Counter()

    def filter_missing_files(self, id_list):
        """Remove IDs whose files are missing."""

        valid_ids = []
        for img_id in id_list:
            img_path = os.path.join(self.images_dir, f"{img_id}.tif")
            if os.path.exists(img_path):
                valid_ids.append(img_id)
        return valid_ids

    def __getitem__(self, index):
        """Load image and corresponding mask."""

        img_id = self.id_list[index]
        img_path = self.find_image_file(img_id)
        if img_path is None:
            raise FileNotFoundError(f"No file found for ID {img_id}")

        image = self.read_image(img_path)
        mask = self.read_mask(img_id)

        if self.verbose:
            print(f"Before processing - Mask shape: {mask.shape}")

        for i in range(mask.shape[-1]):
            count = np.sum(mask[..., i])
            self.class_counts[float(i)] += count

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if self.verbose:
                print(f"After augmentation - Mask shape: {mask.shape}")

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if self.verbose:
                print(f"After preprocessing - Mask shape: {mask.shape}")

        return image, mask

    def __len__(self):
        return len(self.id_list)

    def find_image_file(self, img_id):
        """Find image with supported extensions."""

        extensions = ['.jpg', '.png', '.tif']
        for ext in extensions:
            img_path = os.path.join(self.images_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                return img_path

    def find_mask_file(self, img_id):
        """Find corresponding mask file."""

        extensions = ['.png', ' - Copy.png', '.tif.png.png', '1.bmp.png']
        for ext in extensions:
            mask_path = os.path.join(self.output_dir, f"{img_id}{ext}")
            if os.path.exists(mask_path):
                return mask_path

    def read_image(self, img_path):
        """Load image and convert to RGB."""

        image_pil = Image.open(img_path).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    def read_mask(self, img_id):
        """Load and process segmentation mask."""

        mask_path = self.find_mask_file(img_id)
        if mask_path is None:
            raise FileNotFoundError(f"Mask file not found for ID {img_id}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask from {mask_path}")

        mask_labels = np.vectorize(self.PIXEL_TO_LABEL.get)(mask)
        masks = [(mask_labels == cls) for cls in self.class_values]
        if not masks:
            raise ValueError("No valid masks generated from class_values")
        mask = np.stack(masks, axis=-1).astype('float')
        if mask.shape[-1] != len(self.class_values):
            raise ValueError(f"Expected {len(self.class_values)} channels, got {mask.shape[-1]}")
        return mask / np.max(mask)
   
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """This function resizes the images and masks and convert them to a tensor"""
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Resize(height=512, width=512, always_apply=True),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(test_transform)


# =========================
# Model Creation
# =========================

def create_segmentation_model(encoder='vgg11', encoder_weights='imagenet', 
                              classes=['Pores', 'Open Pores', 'Weld'], 
                              activation='sigmoid', in_channels=3):
    """
    Create UNet segmentation model.

    Args:
        encoder (str): Backbone encoder
        encoder_weights (str): Pretrained weights
        classes (list): Output classes
        activation (str): Activation function
        in_channels (int): Input channels

    Returns:
        torch.nn.Module
    """
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if the specified encoder is valid
    valid_encoders = smp.encoders.get_encoder_names()
    if encoder not in valid_encoders:
        raise ValueError(f"Invalid encoder '{encoder}'. Available encoders: {valid_encoders}")

    # Check if the specified activation function is valid
    valid_activations = ['sigmoid', 'softmax2d', None]
    if activation not in valid_activations:
        raise ValueError(f"Invalid activation '{activation}'. Choose from: {valid_activations}")

    # Create the segmentation model
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation=activation,
        in_channels=in_channels
    )
    
    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(device)

    return model

# =========================
# Loss Function
# =========================

class CustomDiceLoss(smp.utils.losses.DiceLoss):
    """
    Custom loss combining Dice Loss + Cross Entropy.

    Used for better segmentation performance.
    """
    def __init__(self, num_classes, class_weights=None, device='cpu', ignore_index=-1, *args, **kwargs):
        super(CustomDiceLoss, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.device = device
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)

    def forward(self, y_pred, y_true, training=True):
        # Compute Dice loss
        dice_loss = super(CustomDiceLoss, self).forward(y_pred, y_true)
        # Compute cross-entropy loss for logging
        ce_loss = self.ce_loss(y_pred, y_true.argmax(dim=1))
        # Compute total loss for logging (not used for optimization)
        total_loss = dice_loss + ce_loss
        return total_loss, dice_loss, ce_loss

    
# Function to save the best model
def save_best_model(model, path):
    torch.save(model.state_dict(), path)

def get_best_trial_for_encoder(study, encoder_name):
    """Find the best trial number for a specific encoder."""
    best_trial = None
    best_value = float('inf')  # Assuming you're minimizing the objective
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.params.get('encoder') == encoder_name:
            if trial.value < best_value:
                best_value = trial.value
                best_trial = trial
    
    if best_trial is None:
        raise ValueError(f"No completed trial found for encoder: {encoder_name}")
    
    return best_trial.number


def load_best_model(encoder_name, best_trial_number, device, BEST_MODEL_DIR, num_classes):
    """Load the best model for a given encoder based on the best trial."""
    # Construct the specific file name for the best model

    best_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_{encoder_name}_trial_{best_trial_number}.pt")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found at: {best_model_path}")
    
    # Initialize the model
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        classes=num_classes,
        activation='sigmoid',
        in_channels=3
    ).to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model

# =========================
# Trainer Class
# =========================

class SegmentationTrainer:
    """
    A trainer class for performing segmentation model training and validation.
    This class handles data preprocessing, training, validation, and performance
    evaluation using IoU and Dice scores.
    """
    
    def __init__(self, encoder_name, model, optimizer, loss_fn, num_epochs, classes, train_loader, val_loader, csv_filename, device):
        """
        Initializes the trainer with model, optimizer, loss function, data loaders, and other configurations.

        Args:
            encoder_name (str): Name of the encoder used in the segmentation model.
            model (torch.nn.Module): The segmentation model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            loss_fn (callable): Loss function for model training.
            num_epochs (int): Number of training epochs.
            classes (list): List of class names in the segmentation task.
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
            csv_filename (str): Name of the CSV file for logging results.
            device (torch.device): Device to run the training (CPU/GPU).
        """
        self.encoder_name = encoder_name
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.csv_filename = csv_filename
        self.device = device

    def preprocess_data(self, image, mask):
        return image.to(device=self.device), mask.float().to(device=self.device)

    def calculate_per_class_scores(self, predictions, mask, threshold=0.5):
        class_iou_scores = []
        class_dice_scores = []
        for class_idx in range(mask.size(1)):
            class_mask = mask[:, class_idx, :, :]
            if torch.sum(class_mask) > 0:
                class_predictions = predictions[:, class_idx]
                class_predictions_int = class_predictions.round().long()
                class_mask_int = class_mask.round().long()

                tp, fp, fn, tn = smp.metrics.get_stats(class_predictions_int, class_mask_int, mode='multilabel', threshold=threshold)
                class_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
                class_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                class_iou_scores.append(class_iou_score)
                class_dice_scores.append(1 - class_f1_score)
            else:
                zero_score = torch.tensor(0.0, device=self.device)
                class_iou_scores.append(zero_score)
                class_dice_scores.append(zero_score)
        return class_iou_scores, class_dice_scores

    def calculate_overall_scores(self, predictions, mask, threshold=0.5):
        mask_int = mask.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(predictions.round(), mask_int, mode='multilabel', threshold=threshold)
        overall_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        overall_dice_loss = 1 - smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        return overall_iou_score, overall_dice_loss

    def safe_mean(self, scores):
        if not isinstance(scores, list):
            scores = [scores]
        valid_scores = [score.cpu().item() if torch.is_tensor(score) else score for score in scores if score is not None]
        return np.mean(valid_scores) if valid_scores else float('nan')

    def train_one_epoch(self):
        self.model.train()
        train_loop = tqdm(self.train_loader, desc="Training")
        train_class_iou_scores, train_class_dice_scores = [], []
        train_overall_iou_scores, train_overall_dice_losses = [], []
        # Lists to store per-class IoU scores for each batch (for box plot)
        train_per_class_iou_scores = [[] for _ in range(len(self.classes))]

        for image, mask in train_loop:
            image, mask = self.preprocess_data(image, mask)
            predictions = self.model(image)
            total_loss, dice_loss, ce_loss = self.loss_fn(predictions, mask)
                
            class_iou, class_dice = self.calculate_per_class_scores(predictions, mask)
            train_class_iou_scores.extend(class_iou)
            train_class_dice_scores.extend(class_dice)
                
            overall_iou, overall_dice = self.calculate_overall_scores(predictions, mask)
            train_overall_iou_scores.append(overall_iou)
            train_overall_dice_losses.append(overall_dice)

            # Collect per-class IoU scores for box plot
            for class_idx, iou_score in enumerate(class_iou):
                train_per_class_iou_scores[class_idx].append(iou_score.cpu().item() if torch.is_tensor(iou_score) else iou_score)

            self.optimizer.zero_grad()
            dice_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            print(f"Training Batch - Total Loss: {total_loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}")
            train_loop.set_postfix(loss=dice_loss.item())

        self.model.eval()
        val_loop = tqdm(self.val_loader, desc="Validation")
        val_class_iou_scores, val_class_dice_scores = [], []
        val_overall_iou_scores, val_overall_dice_losses = [], []
        # Lists to store per-class IoU scores for each batch (for box plot)
        val_per_class_iou_scores = [[] for _ in range(len(self.classes))]

        with torch.no_grad():
            for image, mask in val_loop:
                image, mask = self.preprocess_data(image, mask)
                predictions = self.model(image)
                total_loss, dice_loss, ce_loss = self.loss_fn(predictions, mask)
                
                class_iou, class_dice = self.calculate_per_class_scores(predictions, mask)
                val_class_iou_scores.extend(class_iou)
                val_class_dice_scores.extend(class_dice)
                
                overall_iou, overall_dice = self.calculate_overall_scores(predictions, mask)
                val_overall_iou_scores.append(overall_iou)
                val_overall_dice_losses.append(overall_dice)

                # Collect per-class IoU scores for box plot
                for class_idx, iou_score in enumerate(class_iou):
                    val_per_class_iou_scores[class_idx].append(iou_score.cpu().item() if torch.is_tensor(iou_score) else iou_score)

                print(f"Validation Batch - Total Loss: {total_loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}")
                val_loop.set_postfix(loss=dice_loss.item())

        # Return 10 values, including per-class IoU scores for plotting
        return (train_class_iou_scores, train_class_dice_scores, train_overall_dice_losses, 
                val_overall_dice_losses, val_class_iou_scores, val_class_dice_scores, 
                train_overall_iou_scores, val_overall_iou_scores, 
                train_per_class_iou_scores, val_per_class_iou_scores)
    
class TestDataset(Dataset):
    """
    Custom dataset for loading test images and corresponding masks.

    Attributes:
        CLASSES (list): List of class names.
        PIXEL_TO_LABEL (dict): Mapping of pixel values to class labels.
        output_dir (str): Directory containing the mask images.
        id_list (list): List of image IDs to load.
        images_dir (str): Directory containing the images.
        classes (list): List of class indices to include.
        class_values (list): List of selected class names.
        augmentation (callable, optional): Augmentation transformations.
        preprocessing (callable, optional): Preprocessing transformations.
    """
    CLASSES = ['Background', 'Pores', 'Open Pores', 'Weld']
    PIXEL_TO_LABEL = {60: 'Background', 78: 'Open Pores', 185: 'Pores', 216: 'Weld'}

    def __init__(self, images_dir, id_list, output_dir=None, augmentation=None, classes=None, preprocessing=None):
        """
        Initializes the dataset.
        Args:
            images_dir (str): Path to the directory containing images.
            id_list (list): List of image IDs.
            output_dir (str, optional): Path to the directory containing masks.
            augmentation (callable, optional): Data augmentation function.
            classes (list, optional): List of class names to include.
            preprocessing (callable, optional): Data preprocessing function.
        """
        self.output_dir = output_dir
        self.id_list = id_list
        self.images_dir = images_dir
        self.classes = [self.CLASSES.index(cls) for cls in classes]
        self.class_values = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        img_id = self.id_list[index]
        img_path = os.path.join(self.images_dir, f"{img_id}.tif")
        
        image = self.read_image(img_path)
        mask = self.read_mask(img_id)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.id_list)

    def read_image(self, img_path):
        image_pil = Image.open(img_path).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    def read_mask(self, img_id):
        mask_path_copy = os.path.join(self.output_dir, f"{img_id} - Copy.png")
        mask_path_normal = os.path.join(self.output_dir, f"{img_id}.png")
        
        if os.path.exists(mask_path_copy):
            mask_path = mask_path_copy
        elif os.path.exists(mask_path_normal):
            mask_path = mask_path_normal
        else:
            raise FileNotFoundError(f"Mask file not found for ID {img_id}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_labels = np.vectorize(self.PIXEL_TO_LABEL.get)(mask)
        masks = [(mask_labels == cls) for cls in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        return mask / np.max(mask)
    
def to_tensor(x, **kwargs):
    """
    Converts an image or mask to a PyTorch tensor format.

    Args:
        x (np.ndarray): Image or mask array.
    
    Returns:
        np.ndarray: Tensor-like array with shape (C, H, W).
    """
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """
    Applies preprocessing transformations including resizing and conversion to tensor.

    Args:
        preprocessing_fn (callable): A preprocessing function for normalization.
    
    Returns:
        albu.Compose: Preprocessing transformations.
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Resize(height=512, width=512, always_apply=True),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation():
    """
    Applies validation augmentation transformations.

    Returns:
        albu.Compose: Validation transformations.
    """
    test_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(test_transform)

def calculate_per_class_scores(predictions, mask, threshold=0.5):
    """
    Calculates per-class IoU scores.
    """
    class_iou_scores = []
    for class_idx in range(mask.size(1)):
        class_mask = mask[:, class_idx, :, :]
        if torch.sum(class_mask) > 0:
            class_predictions = predictions[:, class_idx]
            #class_predictions_int = class_predictions.round().long()
            class_predictions_int = (class_predictions > threshold).long()
            class_mask_int = class_mask.round().long()

            tp, fp, fn, tn = smp.metrics.get_stats(class_predictions_int, class_mask_int, mode='multilabel', threshold=threshold)
            class_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
            class_iou_scores.append(class_iou_score.cpu().numpy())  
        else:
            class_iou_scores.append(0.0)
    return class_iou_scores

def plot_iou_scores_boxplot(iou_scores_by_encoder, encoders, CLASSES, Best_model_dir):
    """
    Plots a boxplot of IoU scores for each class grouped by encoder.
    """
    # Number of classes and encoders
    num_classes = len(CLASSES)
    num_encoders = len(encoders)
    bar_width = 0.15  # Adjust this value to control box width
    
    # Prepare the positions for each encoder within each class
    class_positions = np.arange(num_classes) * (num_encoders + 1)  # Space between classes
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a list of all scores for each class (grouped by encoder)
    all_scores = []
    for idx, class_name in enumerate(CLASSES):

        class_scores = [iou_scores_by_encoder[encoder][idx] for encoder in encoders]
        all_scores.append(class_scores)
    
    # Create box plots for each class (grouped by encoder)
    ax.boxplot(all_scores, positions=class_positions, widths=bar_width, patch_artist=True)
    
    # Set axis labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Average IoU Score')
    ax.set_title('Average IoU Scores by Encoder and Class')
    
    # Set x-axis tick positions in the middle of each encoder group
    tick_positions = class_positions + bar_width * (num_encoders - 1) / 2
    ax.set_xticks(tick_positions)  # Set ticks in the middle of each class group
    ax.set_xticklabels(CLASSES)  # Set class names as tick labels
    
    # Set y-axis limits
    ax.set_ylim(0, 1)  # Assuming IoU score is between 0 and 1, adjust if necessary
    
    # Add a legend for the encoders (corresponding to each box in the class group)
    ax.legend([plt.Line2D([0], [0], color='black', lw=2)] * num_encoders, encoders, title="Encoders", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot and display it
    plt.tight_layout()
    plt.savefig(Best_model_dir + 'test_plot_optuna_boxplot.png')
    plt.show()

def plot_iou_scores_by_encoder(iou_scores_by_encoder, encoders, CLASSES, best_model_dir):
    # Number of classes and encoders
    num_classes = len(CLASSES)
    
    # Create a plot for each encoder
    for encoder in encoders:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_scores = []
        for idx in range(num_classes):
            scores = iou_scores_by_encoder[encoder][idx]  
            
            # Ensure scores is a list (even for a single score)
            if not isinstance(scores, list):
                scores = [scores]
            
            all_scores.append(scores)
        # Check if the lengths match
        if len(all_scores) != num_classes:
            print(f"Mismatch in scores for encoder {encoder}. Skipping plot.")
            continue
        
        # Set positions for the boxplots
        positions = np.arange(num_classes)
        
        # Plot the boxplot
        ax.boxplot(all_scores, positions=positions, widths=0.5, patch_artist=True)
        
        # Set labels and title
        ax.set_xticks(positions)
        ax.set_xticklabels(CLASSES)
        ax.set_xlabel('Classes')
        ax.set_ylabel('IoU Scores')
        ax.set_title(f'IoU Scores by Encoder - {encoder}')
        # Save the plot with the encoder's name as the file name
        plot_filename = f'{encoder}_iou_scores.png'
        plt.tight_layout()
        plt.savefig(best_model_dir + plot_filename)
        plt.close() 

def visualize_predictions(encoders, test_image_tensor, CLASSES, DEVICE, BEST_MODEL_DIR):
    """
    Visualizes the predictions for each encoder using the best model.
    
    Args:
    - encoders (list): List of encoder names.
    - test_image_tensor (torch.Tensor): Input test image tensor.
    - CLASSES (list): List of class names.
    - DEVICE (str): Device to use ('cuda' or 'cpu').
    - BEST_MODEL_DIR (str): Directory containing the best models.
    """

    num_classes = len(CLASSES)
    fig, axes = plt.subplots(num_classes, len(encoders), figsize=(15, 5 * num_classes))

    # Load Optuna study
    study = optuna.load_study(study_name="Segmentation_20", storage="sqlite:///optuna_study_with_weights.db")

    for i, encoder_name in enumerate(encoders):
        # Get the best trial number for the encoder
        best_trial_number = get_best_trial_for_encoder(study, encoder_name)
        print(f"Loading best model for {encoder_name}, Trial Number: {best_trial_number}")

        # Load the best model
        model = load_best_model(encoder_name, best_trial_number, DEVICE, BEST_MODEL_DIR, num_classes)
        model.eval()

        # Get predictions
        with torch.no_grad():
            pr_mask = model(test_image_tensor.to(DEVICE))
            pr_mask = (pr_mask.squeeze().cpu().numpy() > 0.3).astype(np.uint8)
            pr_mask = remove_small_objects(pr_mask.astype(bool), min_size=50).astype(np.uint8)

        # Plot results
        for j in range(num_classes):
            if num_classes == 1:
                ax = axes[i] if len(encoders) > 1 else axes
            else:
                ax = axes[j, i] if num_classes > 1 else axes[i]

            ax.imshow(pr_mask[j] if num_classes > 1 else pr_mask, cmap='viridis')
            ax.set_title(f"{encoder_name} - {CLASSES[j]}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(BEST_MODEL_DIR + 'test_optuna.png')
    plt.show()

def plot_segmentation_masks(encoders, study, CLASSES, test_image_tensors, BEST_MODEL_DIR, DEVICE, load_best_model, get_best_trial_for_encoder, num_classes, figsize=(16, 3), save_dir='/home/sonawane/backup/src/final_trial/'):
    """
    Create subplots to display segmentation masks for multiple test images across different encoders and classes.

    Returns:
        None: Saves the plots for each encoder and displays them (optional).
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over the encoders
    for i, encoder_name in enumerate(encoders):
        best_trial_number = get_best_trial_for_encoder(study, encoder_name)
        print(f"Loading best model for {encoder_name}, Trial Number: {best_trial_number}")

        # Load the model
        model = load_best_model(encoder_name, best_trial_number, DEVICE, BEST_MODEL_DIR, num_classes)
        if model is None:
            print(f"Skipping {encoder_name} due to missing model.")
            continue

        # Create subplots for the current encoder
        fig, axes = plt.subplots(len(test_image_tensors), num_classes, figsize=(figsize[0], len(test_image_tensors) * figsize[1]))

        # Iterate over test images
        for idx, test_image_tensor in enumerate(test_image_tensors):
            with torch.no_grad():
                pr_mask = model(test_image_tensor.to(DEVICE))
                pr_mask = (pr_mask.squeeze().cpu().numpy() > 0.3).astype(np.uint8)
                pr_mask = remove_small_objects(pr_mask.astype(bool), min_size=50).astype(np.uint8)

            # Plot results for each class
            for j in range(num_classes):
                ax = axes[idx, j] if num_classes > 1 else axes[idx]
                ax.imshow(pr_mask[j], cmap='viridis')
                ax.set_title(f"{encoder_name} - {CLASSES[j]}")
                ax.axis('off')

        # Adjust layout and save the image for the current encoder
        plt.tight_layout()
        output_path = os.path.join(save_dir, f'test_images_{encoder_name}.png')
        plt.savefig(output_path)
        plt.close(fig)

def plot_iou_scores_by_encoder(encoders, iou_scores_by_encoder, CLASSES, BEST_MODEL_DIR, bar_width=0.15, figsize=(12, 8)):
    """
    Create a bar plot of average IoU scores by encoder and class.
    
    Args:
        encoders (list): List of encoder names (e.g., ['vgg11', 'resnet18', ...]).
        iou_scores_by_encoder (dict): Dictionary with encoder names as keys and lists of IoU scores per class as values.
        CLASSES (list): List of class names (e.g., ['Pores', 'Open Pores', 'Weld']).
        BEST_MODEL_DIR (str): Directory path to save the plot.
        bar_width (float, optional): Width of the bars. Defaults to 0.15.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 8).
    
    Returns:
        None: Saves the plot to BEST_MODEL_DIR/test_plot_optuna.png and displays it.
    """
    num_encoders = len(encoders)
    num_classes = len(CLASSES)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define index positions for encoders
    index = np.arange(num_encoders)
    class_positions = [index + bar_width * i for i in range(num_classes)]
    
    # Plot bars for each class
    for idx, class_name in enumerate(CLASSES):
        scores = [iou_scores_by_encoder[encoder][idx] for encoder in encoders]
        ax.bar(class_positions[idx], scores, bar_width, label=class_name)
    
    # Customize the plot
    ax.set_xlabel('Encoders')
    ax.set_ylabel('Average IoU Score')
    ax.set_title('Average IoU Scores by Encoder and Class')
    ax.set_xticks(index + bar_width * (num_classes - 1) / 2)
    ax.set_xticklabels(encoders)
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save and display the plot
    output_path = os.path.join(BEST_MODEL_DIR, 'test_plot_optuna.png')
    plt.savefig(output_path)
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(dataset):
    if isinstance(dataset, Subset):
        return analyze_class_distribution(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        class_areas = np.zeros(len(dataset.datasets[0].CLASSES))
        for sub_dataset in dataset.datasets:
            sub_proportions = analyze_class_distribution(sub_dataset)
            sub_total = sum([np.sum(mask) for _, mask in sub_dataset])
            class_areas += sub_proportions * sub_total
        total_area = np.sum(class_areas)
        return class_areas / total_area if total_area > 0 else class_areas

    class_areas = np.zeros(len(dataset.CLASSES))
    for i, (_, mask) in enumerate(tqdm(dataset, desc="Analyzing class distribution")):
        if len(mask.shape) != 3:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        if mask.shape[0] == len(dataset.CLASSES):  # (num_classes, height, width)
            class_areas += np.sum(mask, axis=(1, 2))
        elif mask.shape[-1] == len(dataset.CLASSES):  # (height, width, num_classes)
            class_areas += np.sum(mask, axis=(0, 1))
        else:
            raise ValueError(f"Mask channels ({mask.shape[0]} or {mask.shape[-1]}) != num_classes ({len(dataset.CLASSES)})")
    total_area = np.sum(class_areas)
    return class_areas / total_area if total_area > 0 else class_areas

def calculate_class_weights(dataset, classes):
    class_proportions = analyze_class_distribution(dataset)
    print("Class Proportions:", dict(zip(classes, class_proportions)))

    initial_weights = np.where(class_proportions > 0, 1.0 / class_proportions, float('inf'))
    min_weight = np.min(initial_weights[initial_weights != float('inf')]) if np.any(initial_weights != float('inf')) else 1.0
    initial_weights = np.where(initial_weights == float('inf'), len(classes), initial_weights / min_weight)

    class_weights = initial_weights / np.max(initial_weights) * 5.0
    class_weights = np.clip(class_weights, 1.0, 5.0)

    return initial_weights.tolist(), class_weights.tolist()


# Global dictionary to store loss history
loss_history = {}

def objective(trial, train_loader, val_loader, test_dataloader, num_classes, class_weights, CLASSES, DEVICE, BEST_MODEL_DIR, NUM_EPOCHS, encoder):
    # Debug print to verify the encoder value
    print(f"Objective called with encoder: {encoder}")

    # Validate encoder
    valid_encoders = ['vgg11', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if encoder not in valid_encoders:
        raise ValueError(f"Invalid encoder: {encoder}. Must be one of {valid_encoders}")

    # Hyperparameters
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    # Set encoder as a user attribute for filtering later
    trial.set_user_attr('encoder', encoder)
    trial.set_user_attr('trial_number', trial.number)

    # Create model with encoder-specific preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    model = smp.Unet(
        encoder_name=encoder, encoder_weights='imagenet', in_channels=3,
        classes=num_classes, activation='sigmoid'
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    loss_fn = CustomDiceLoss(num_classes=num_classes, class_weights=class_weights_tensor, device=DEVICE, ignore_index=-1)

    trainer = SegmentationTrainer(
        encoder_name=encoder, model=model, optimizer=optimizer, loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS, classes=CLASSES, train_loader=train_loader,
        val_loader=val_loader, csv_filename=None, device=DEVICE
    )

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    previous_iou_plot_path = None
    best_iou_plot_path = None
    train_losses = []
    val_losses = []
    best_train_per_class_iou_scores = None
    best_val_per_class_iou_scores = None

    # CSV file for logging IoU scores and Dice losses per epoch
    iou_csv_path = os.path.join(BEST_MODEL_DIR, f'iou_scores_{encoder}_trial_{trial.number}.csv')
    with open(iou_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Header: Epoch, Train IoU per class, Val IoU per class, Train Dice Loss, Val Dice Loss
        header = ['Epoch'] + [f'Train_IoU_{cls}' for cls in CLASSES] + [f'Val_IoU_{cls}' for cls in CLASSES] + ['Train_Dice_Loss', 'Val_Dice_Loss']
        csv_writer.writerow(header)

        for epoch in range(NUM_EPOCHS):
            (train_class_iou_scores, train_class_dice_scores, train_overall_dice_losses, 
             val_overall_dice_losses, val_class_iou_scores, val_class_dice_scores, 
             train_overall_iou_scores, val_overall_iou_scores, 
             train_per_class_iou_scores, val_per_class_iou_scores) = trainer.train_one_epoch()
            
            avg_train_dice_loss = np.mean([loss.item() for loss in train_overall_dice_losses])
            avg_val_dice_loss = np.mean([loss.item() for loss in val_overall_dice_losses])
            
            train_losses.append(avg_train_dice_loss)
            val_losses.append(avg_val_dice_loss)

            # Calculate mean IoU scores per class
            train_mean_iou = [np.mean(scores) if scores else 0.0 for scores in train_per_class_iou_scores]
            val_mean_iou = [np.mean(scores) if scores else 0.0 for scores in val_per_class_iou_scores]

            # Log IoU scores and Dice losses to CSV
            row = [epoch] + train_mean_iou + val_mean_iou + [avg_train_dice_loss, avg_val_dice_loss]
            csv_writer.writerow(row)

            if avg_val_dice_loss < best_val_loss and not np.isnan(avg_val_dice_loss):
                best_val_loss = avg_val_dice_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                if previous_iou_plot_path and os.path.exists(previous_iou_plot_path):
                    os.remove(previous_iou_plot_path)
                    print(f"Deleted previous IoU plot: {previous_iou_plot_path}")
                previous_iou_plot_path = best_iou_plot_path
                best_iou_plot_path = os.path.join(BEST_MODEL_DIR, f'iou_bar_graph_{encoder}_trial_{trial.number}_epoch_{epoch}.png')
                best_train_per_class_iou_scores = train_per_class_iou_scores
                best_val_per_class_iou_scores = val_per_class_iou_scores

    trial.set_user_attr('best_epoch', best_epoch)
    trial.set_user_attr('best_iou_plot_path', best_iou_plot_path)

    # Store the loss history for this trial
    trial_key = f"{encoder}_trial_{trial.number}"
    loss_history[trial_key] = {'train': train_losses, 'val': val_losses}

    trial_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_{encoder}_trial_{trial.number}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, trial_model_path)

    return best_val_loss

# Visualization: Combined function for all visualizations
def plot_all_visualizations(encoders, trained_models, test_dice_losses_by_encoder, iou_scores_by_encoder, test_image_path, test_image_dir, CLASSES, DEVICE, BEST_MODEL_DIR, best_trials, studies):
    """
    Consolidated function to generate all visualizations for the thesis.
    
    Args:
        encoders (list): List of encoder names.
        trained_models (dict): Dictionary mapping encoder names to trained models.
        test_dice_losses_by_encoder (dict): Dictionary mapping encoder names to lists of test Dice losses.
        iou_scores_by_encoder (dict): Dictionary mapping encoder names to lists of per-class IoU scores.
        test_image_path (str): Path to a single test image for visualization.
        test_image_dir (str): Directory containing test images for random sampling.
        CLASSES (list): List of class names.
        DEVICE (str): Device to use ('cuda' or 'cpu').
        BEST_MODEL_DIR (str): Directory to save the plots.
        best_trials (dict): Dictionary mapping encoder names to their best trial numbers.
        study (optuna.Study): Optuna study object for accessing trial data.
    """
    num_classes = len(CLASSES)

    # Transformations for test images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load single test image
    test_image = Image.open(test_image_path).convert('RGB')
    test_image_tensor = transform(test_image).unsqueeze(0).to(DEVICE)
    print(f"Test image tensor shape: {test_image_tensor.shape}, mean: {test_image_tensor.mean().item()}, std: {test_image_tensor.std().item()}")

    # Load 5 random test images
    test_image_paths = random.sample([os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir)], 5)
    test_images = [Image.open(img_path).convert('RGB') for img_path in test_image_paths]
    test_image_tensors = [transform(img).unsqueeze(0).to(DEVICE) for img in test_images]

    # 1. Per-Class Test IoU Bar Graph (Main Body)
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(encoders))
    class_positions = [index + bar_width * i for i in range(num_classes)]
    for idx, class_name in enumerate(CLASSES):
        scores = [iou_scores_by_encoder[encoder][idx] for encoder in encoders]
        ax.bar(class_positions[idx], scores, bar_width, label=class_name)
    ax.set_xlabel('Encoders')
    ax.set_ylabel('Mean IoU Score')
    ax.set_title('Mean IoU Scores by Encoder and Class (Test Set)')
    ax.set_xticks(index + bar_width * (num_classes - 1) / 2)
    ax.set_xticklabels(encoders)
    ax.set_ylim(0, 1)
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'test_iou_per_class_bar_graph.png'))
    plt.close()

    # 2. Per-Class Test Dice Loss Bar Graph (Main Body)
    # Calculate per-class Dice loss from test_dice_losses_by_encoder
    per_class_dice_losses = {}
    for encoder in encoders:
        if encoder not in test_dice_losses_by_encoder:
            continue
        # Since Dice loss isn't per-class in your current setup, we'll approximate it by evaluating the model on the test set
        # For simplicity, we'll use the average Dice loss per class by re-evaluating (this is a placeholder; ideally, you'd modify your evaluation loop to compute per-class Dice loss)
        # Here, we'll assume Dice loss is uniform across classes for simplicity (you should modify your evaluation to compute per-class Dice loss if needed)
        avg_dice_loss = np.mean(test_dice_losses_by_encoder[encoder])
        per_class_dice_losses[encoder] = [avg_dice_loss] * num_classes  # Placeholder; replace with actual per-class Dice loss if available

    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, class_name in enumerate(CLASSES):
        losses = [per_class_dice_losses[encoder][idx] for encoder in encoders if encoder in per_class_dice_losses]
        valid_encoders = [encoder for encoder in encoders if encoder in per_class_dice_losses]
        ax.bar([i + bar_width * idx for i in range(len(valid_encoders))], losses, bar_width, label=class_name)
    ax.set_xlabel('Encoders')
    ax.set_ylabel('Mean Dice Loss')
    ax.set_title('Mean Dice Loss by Encoder and Class (Test Set)')
    ax.set_xticks([i + bar_width * (num_classes - 1) / 2 for i in range(len(valid_encoders))])
    ax.set_xticklabels(valid_encoders)
    ax.set_ylim(0, 1)
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'test_dice_loss_per_class_bar_graph.png'))
    plt.close()

    # 3. Class Distribution and Weights Plot (Main Body)
    # Class proportions and weights should be passed to this function or recomputed here
    # For now, we'll assume they are available from calculate_class_weights (you may need to modify main() to pass them)
    # Placeholder data (replace with actual class_proportions and class_weights from calculate_class_weights)
    class_proportions = np.array([0.1, 0.05, 0.85])  # Example proportions for Pores, Open Pores, Weld
    class_weights = np.array([5.0, 5.0, 1.0])  # Example weights

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(CLASSES))
    ax1.bar(x - bar_width/2, class_proportions, bar_width, label='Class Proportion', color='skyblue')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Proportion', color='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.bar(x + bar_width/2, class_weights, bar_width, label='Class Weight', color='lightcoral')
    ax2.set_ylabel('Weight', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    ax2.set_ylim(0, max(class_weights) + 1)

    fig.suptitle('Class Distribution and Weights')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'class_distribution_and_weights.png'))
    plt.close()

    # 4. Hyperparameter Impact Plot (Main Body)
    # Scatter plot of learning rate vs weight decay vs best validation loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for encoder in encoders:
        if encoder not in studies:
            continue
        study = studies[encoder]
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            continue
        lrs = [t.params['lr'] for t in trials]
        wds = [t.params['weight_decay'] for t in trials]
        losses = [t.value for t in trials]
        scatter = ax.scatter(lrs, wds, c=losses, cmap='viridis', label=encoder, s=100)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Weight Decay')
    ax.set_title('Hyperparameter Impact on Best Validation Loss')
    plt.colorbar(scatter, label='Best Validation Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'hyperparameter_impact.png'))
    plt.close()

    # 5. Per-Batch Test Dice Loss Plot (Appendix)
    plt.figure(figsize=(12, 6))
    for encoder, dice_losses in test_dice_losses_by_encoder.items():
        if dice_losses:
            batches = range(len(dice_losses))
            plt.plot(batches, dice_losses, label=f'{encoder} Dice Loss')
    plt.xlabel('Batch')
    plt.ylabel('Dice Loss')
    plt.title('Test Dice Loss per Encoder Across Batches (Batch Size = 16, 33 Test Images)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(BEST_MODEL_DIR, "test_dice_losses.png"))
    plt.close()

    # 6. Loss Curves (Main Body)
    for encoder, trial_number in best_trials.items():
        csv_path = os.path.join(BEST_MODEL_DIR, f'iou_scores_{encoder}_trial_{trial_number}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file not found for {encoder}_trial_{trial_number}. Skipping.")
            continue
        df = pd.read_csv(csv_path)
        if 'Train_Dice_Loss' not in df.columns or 'Val_Dice_Loss' not in df.columns:
            print(f"Dice loss columns not found in {csv_path}. Skipping.")
            continue
        train_losses = df['Train_Dice_Loss'].values
        val_losses = df['Val_Dice_Loss'].values
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss', linestyle='dashed', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Loss')
        plt.title(f'Train vs Validation Loss - {encoder} (Trial {trial_number})')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(BEST_MODEL_DIR, f"loss_curve_{encoder}_trial_{trial_number}.png"))
        plt.close()

    # 7. IoU Trends (Main Body)
    for encoder, trial_number in best_trials.items():
        csv_path = os.path.join(BEST_MODEL_DIR, f'iou_scores_{encoder}_trial_{trial_number}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file not found for {encoder}_trial_{trial_number}. Skipping.")
            continue
        df = pd.read_csv(csv_path)
        epochs = df['Epoch'].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for cls in CLASSES:
            train_iou_col = f'Train_IoU_{cls}'
            if train_iou_col in df.columns:
                ax1.plot(epochs, df[train_iou_col], label=f'{cls}')
        ax1.set_title(f'Training IoU Scores - {encoder} (Trial {trial_number})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('IoU Score')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True)
        for cls in CLASSES:
            val_iou_col = f'Val_IoU_{cls}'
            if val_iou_col in df.columns:
                ax2.plot(epochs, df[val_iou_col], label=f'{cls}')
        ax2.set_title(f'Validation IoU Scores - {encoder} (Trial {trial_number})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BEST_MODEL_DIR, f"iou_trends_{encoder}_trial_{trial_number}.png"))
        plt.close()
    
    # 7.5. Per-Class IoU Bar Graph at Best Epoch 
    for encoder, trial_number in best_trials.items():
        # Read the CSV file for this trial
        csv_path = os.path.join(BEST_MODEL_DIR, f'iou_scores_{encoder}_trial_{trial_number}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file not found for {encoder}_trial_{trial_number}. Skipping.")
            continue
        
        # Load the CSV data
        df = pd.read_csv(csv_path)
        
        # Get the best epoch from the trial's user attributes
        trial = [t for t in study.trials if t.number == trial_number][0]
        best_epoch = trial.user_attrs.get('best_epoch', -1)
        if best_epoch == -1:
            print(f"No best epoch found for {encoder}_trial_{trial_number}. Skipping.")
            continue
        
        # Extract IoU scores at the best epoch
        df_best_epoch = df[df['Epoch'] == best_epoch]
        if df_best_epoch.empty:
            print(f"No data for best epoch {best_epoch} in {csv_path}. Skipping.")
            continue
        
        train_per_class_iou_scores = []
        val_per_class_iou_scores = []
        for cls in CLASSES:
            train_iou_col = f'Train_IoU_{cls}'
            val_iou_col = f'Val_IoU_{cls}'
            if train_iou_col in df_best_epoch.columns and val_iou_col in df_best_epoch.columns:
                train_score = df_best_epoch[train_iou_col].iloc[0]
                val_score = df_best_epoch[val_iou_col].iloc[0]
                # Since the scores in the CSV are already means, wrap them in a list for consistency with the function
                train_per_class_iou_scores.append([train_score])
                val_per_class_iou_scores.append([val_score])
            else:
                train_per_class_iou_scores.append([0.0])
                val_per_class_iou_scores.append([0.0])
        
        # Plot the bar graph
        bar_width = 0.35
        x = np.arange(len(CLASSES))
        fig, ax = plt.subplots(figsize=(10, 6))
        train_mean_iou = [np.mean(scores) for scores in train_per_class_iou_scores]
        val_mean_iou = [np.mean(scores) for scores in val_per_class_iou_scores]
        ax.bar(x - bar_width/2, train_mean_iou, bar_width, label='Training IoU', color='skyblue')
        ax.bar(x + bar_width/2, val_mean_iou, bar_width, label='Validation IoU', color='lightcoral')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Mean IoU Score')
        ax.set_title(f'Mean IoU Scores - {encoder} (Best Epoch {best_epoch}, Trial {trial_number})')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=45)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_path = os.path.join(BEST_MODEL_DIR, f'iou_bar_graph_best_epoch_{encoder}_trial_{trial_number}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved IoU bar graph for {encoder}_trial_{trial_number} at best epoch {best_epoch} to {output_path}")

    # 8. IoU Scores Boxplot (Appendix)
    num_classes = len(CLASSES)
    class_positions = np.arange(num_classes) * (len(encoders) + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    all_scores = []
    for idx, class_name in enumerate(CLASSES):
        class_scores = [iou_scores_by_encoder[encoder][idx] for encoder in encoders]
        all_scores.append(class_scores)
    ax.boxplot(all_scores, positions=class_positions, widths=0.5, patch_artist=True)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Average IoU Score')
    ax.set_title('Average IoU Scores by Class (Test Set)')
    tick_positions = class_positions + 0.5 * (len(encoders) - 1) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1)
    ax.legend([plt.Line2D([0], [0], color='black', lw=2)] * len(encoders), encoders, title="Encoders", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'test_iou_boxplot.png'))
    plt.close()

    # 9. IoU Scores Boxplot per Encoder (Appendix)
    for encoder in encoders:
        fig, ax = plt.subplots(figsize=(12, 6))
        all_scores = []
        for idx in range(num_classes):
            scores = iou_scores_by_encoder[encoder][idx]
            if not isinstance(scores, list):
                scores = [scores]
            all_scores.append(scores)
        positions = np.arange(num_classes)
        ax.boxplot(all_scores, positions=positions, widths=0.5, patch_artist=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(CLASSES)
        ax.set_xlabel('Classes')
        ax.set_ylabel('IoU Scores')
        ax.set_title(f'IoU Scores by Encoder - {encoder} (Test Set)')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(BEST_MODEL_DIR, f'{encoder}_iou_scores_boxplot.png'))
        plt.close()

    # 10. Visualize Predictions for Single Test Image (Per Class) (Main Body)
    fig, axes = plt.subplots(num_classes, len(encoders), figsize=(16, num_classes * 3))
    for i, encoder_name in enumerate(encoders):
        if encoder_name not in trained_models:
            continue
        model = trained_models[encoder_name]
        model.eval()
        with torch.no_grad():
            pr_mask = model(test_image_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy().round()
        for j in range(num_classes):
            ax = axes[j, i] if num_classes > 1 else axes[i]
            ax.imshow(pr_mask[j], cmap='viridis')
            ax.set_title(f"{encoder_name} - {CLASSES[j]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'test_single_image_predictions.png'))
    plt.close()

    # 11. Visualize Combined Predictions for Single Test Image (Main Body)
    COLORS = [(255, 0, 0), (255, 255, 0), (0, 255, 0)]
    fig, axes = plt.subplots(1, len(encoders), figsize=(16, 5))
    for i, encoder_name in enumerate(encoders):
        if encoder_name not in trained_models:
            continue
        model = trained_models[encoder_name]
        model.eval()
        with torch.no_grad():
            pr_mask = model(test_image_tensor).squeeze().cpu().numpy()
            combined_mask = np.zeros((pr_mask.shape[1], pr_mask.shape[2], 3), dtype=np.uint8)
            for j in range(num_classes):
                mask = pr_mask[j] > 0.4
                for k in range(3):
                    combined_mask[:, :, k] += (mask * COLORS[j][k]).astype(np.uint8)
        axes[i].imshow(combined_mask)
        axes[i].set_title(encoder_name)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(BEST_MODEL_DIR, 'test_combined_predictions.png'))
    plt.close()

    # 12. Visualize Predictions for 5 Random Test Images (Per Encoder) (Appendix)
    for encoder_name in encoders:
        if encoder_name not in trained_models:
            continue
        model = trained_models[encoder_name]
        model.eval()
        fig, axes = plt.subplots(len(test_image_tensors), num_classes, figsize=(16, len(test_image_tensors) * 3))
        for idx, test_image_tensor in enumerate(test_image_tensors):
            with torch.no_grad():
                pr_mask = model(test_image_tensor).squeeze().cpu().numpy().round()
            for j in range(num_classes):
                ax = axes[idx, j] if num_classes > 1 else axes[idx]
                ax.imshow(pr_mask[j], cmap='viridis')
                ax.set_title(f"{encoder_name} - {CLASSES[j]}")
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(BEST_MODEL_DIR, f'test_5_images_{encoder_name}.png'))
        plt.close()

    # 13. Visualize Predictions for 5 Random Test Images with Original Image (Main Body)
    for encoder_name in encoders:
        if encoder_name not in trained_models:
            continue
        model = trained_models[encoder_name]
        model.eval()
        fig, axes = plt.subplots(len(test_image_tensors), num_classes + 1, figsize=(16, len(test_image_tensors) * 3))
        for idx, (test_image_tensor, test_image) in enumerate(zip(test_image_tensors, test_images)):
            ax = axes[idx, 0]
            ax.imshow(test_image)
            ax.set_title("Original Image")
            ax.axis('off')
            with torch.no_grad():
                pr_mask = model(test_image_tensor).squeeze().cpu().numpy().round()
            for j in range(num_classes):
                ax = axes[idx, j + 1]
                ax.imshow(pr_mask[j], cmap='viridis')
                ax.set_title(f"{encoder_name} - {CLASSES[j]}")
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(BEST_MODEL_DIR, f'test_5_images_with_original_{encoder_name}.png'))
        plt.close()

# =========================
# MAIN FUNCTION
# =========================

def main():
    """
    Entry point of the script.

    Steps:
    1. Load datasets
    2. Create model
    3. Train model
    4. Evaluate performance
    """

    # Set random seeds for reproducibility
    import random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Process datasets
    kop_df = read_csv(KOP_FILE_PATH)
    kop_ids, skipped_kop_files = process_image_ids(kop_df, 'data_row.external_id')
    op_df = read_csv(OP_FILE_PATH)
    op_ids, skipped_op_ids = process_image_ids(op_df, 'data_row/external_id')
    gop_df = read_csv(CSV_FILE_PATH)
    gop_ids, skipped_gop_ids = process_image_ids(gop_df, 'data_row.external_id')
    gkop_image_list = read_ndjson(JSON_FILE_PATH)
    gkop_ids = clean_image_list(gkop_image_list)
    labels_df = read_csv(labels_csv)
    labels_ids = process_new_image_ids(labels_df, 'data_row.external_id', new_labels_dir)

    # Define model parameters
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['Pores', 'Open Pores', 'Weld']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BEST_MODEL_DIR = "/home/sonawane/backup/src/dice_loss_model_7/"
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    encoders = ['vgg11', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    study_name_prefix = "segmentation_study"

    NUM_EPOCHS = 25
    N_TRIALS = 20

    # Define preprocessing function (initially for vgg11 for weights calculation)
    preprocessing_fn = smp.encoders.get_preprocessing_fn('vgg11', ENCODER_WEIGHTS)

    # Create datasets for weight calculation
    train_kop_dataset_weights = SegmentationDataset(
        images_dir=TRAIN_KOP_DIR, id_list=kop_ids, output_dir=MASK_KOP_DIR,
        augmentation=None, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
        verbose=True
    )
    new_labels_dataset_weights = SegmentationDataset(
        images_dir=new_labels_dir, id_list=labels_ids, output_dir=mask_folder,
        augmentation=None, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
        verbose=True
    )
    train_op_dataset_weights = SegmentationDataset(
        images_dir=TRAIN_OP_DIR, id_list=op_ids, output_dir=MASK_OP_DIR,
        augmentation=None, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
        verbose=True
    )
    train_gop_dataset_weights = SegmentationDataset(
        images_dir=TRAIN_GOP_DIR, id_list=gop_ids, output_dir=MASK_GOP_DIR,
        augmentation=None, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
        verbose=True
    )
    train_grezfall_keine_dataset_weights = SegmentationDataset(
        images_dir=TRAIN_GKOP_DIR, id_list=gkop_ids, output_dir=MASK_GKOP_DIR,
        augmentation=None, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
        verbose=True
    )

    all_datasets_weights = [train_kop_dataset_weights, train_op_dataset_weights,
                            train_gop_dataset_weights, train_grezfall_keine_dataset_weights,
                            new_labels_dataset_weights]
    dataset_weights = ConcatDataset(all_datasets_weights)
    train_size_weights = int(0.7 * len(dataset_weights))
    val_size_weights = int(0.2 * len(dataset_weights))
    test_size_weights = len(dataset_weights) - train_size_weights - val_size_weights
    torch.manual_seed(42)
    train_dataset_weights, _, _ = random_split(dataset_weights, [train_size_weights, val_size_weights, test_size_weights])

    # Calculate class weights once
    initial_weights, class_weights = calculate_class_weights(train_dataset_weights, CLASSES)
    print(f"Initial Weights: {initial_weights}, Class Weights: {class_weights}")

    # Create training datasets
    train_kop_dataset = SegmentationDataset(TRAIN_KOP_DIR, kop_ids, MASK_KOP_DIR, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
    new_labels_dataset = SegmentationDataset(new_labels_dir, labels_ids, mask_folder, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
    train_op_dataset = SegmentationDataset(TRAIN_OP_DIR, op_ids, MASK_OP_DIR, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
    train_gop_dataset = SegmentationDataset(TRAIN_GOP_DIR, gop_ids, MASK_GOP_DIR, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
    train_grezfall_keine_dataset = SegmentationDataset(TRAIN_GKOP_DIR, gkop_ids, MASK_GKOP_DIR, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)

    all_datasets = [train_kop_dataset, train_op_dataset, train_gop_dataset, train_grezfall_keine_dataset, new_labels_dataset]
    dataset = ConcatDataset(all_datasets)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    validation_split, test_split = 0.2, 0.1
    val_size = int(validation_split * total_samples)
    test_size = int(test_split * total_samples)
    train_size = total_samples - val_size - test_size
    torch.manual_seed(42)
    train_dataset, val_dataset, test_new_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders with a fixed worker seed for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    num_workers = min(os.cpu_count(), 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Prepare test dataset
    test_id_list = read_csv(TEST_FILE_PATH)['data_row__external_id'].astype(str).tolist()
    test_ids = [filename.rsplit('.', 1)[0] for filename in test_id_list if filename.endswith('.jpg')]
    test_kop_list = load_test_data(TRAIN_KOP_DIR, test_ids)
    test_op_list = load_test_data(TRAIN_OP_DIR, test_ids)
    test_gop_list = load_test_data(TRAIN_GOP_DIR, test_ids)
    test_gkop_list = load_test_data(TRAIN_GKOP_DIR, test_ids)
    all_test_images = test_kop_list + test_op_list + test_gop_list + test_gkop_list

    os.makedirs(TEST_DIR, exist_ok=True)
    for dir_path, image_list in zip([TRAIN_KOP_DIR, TRAIN_OP_DIR, TRAIN_GOP_DIR, TRAIN_GKOP_DIR], [test_kop_list, test_op_list, test_gop_list, test_gkop_list]):
        for image in image_list:
            shutil.copy(os.path.join(dir_path, image), os.path.join(TEST_DIR, image))

    test_dataset = TestDataset(TEST_DIR, test_ids, TEST_MASK_DIR, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    test_dataset = ConcatDataset([test_dataset, test_new_dataset])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=11,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")

    # Load or create Optuna studies for each encoder
    storage_path = f"sqlite:///{BEST_MODEL_DIR}/optuna_study.db"
    
    # Delete the existing database to start fresh
    db_path = os.path.join(BEST_MODEL_DIR, "optuna_study.db")
    if os.path.exists(db_path):
        print(f"Deleting existing Optuna database at {db_path}")
        os.remove(db_path)

    trained_models = {}
    iou_scores_by_encoder = {}
    test_dice_losses_by_encoder = {}
    best_trials = {}  # To store the best trial number for each encoder

    # Dictionary to hold studies for each encoder
    studies = {}

    # Valid encoders for validation
    valid_encoders = ['vgg11', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    for encoder in encoders:
        if encoder not in valid_encoders:
            print(f"Invalid encoder: {encoder}. Skipping.")
            continue

        study_name = f"{study_name_prefix}_{encoder}"
        print(f"Creating new study for {encoder}")
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_path)
        studies[encoder] = study

        # Optimize with N_TRIALS / len(encoders) trials per encoder to distribute trials evenly
        trials_per_encoder = max(1, N_TRIALS // len(encoders))
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, test_dataloader, len(CLASSES), class_weights, CLASSES, DEVICE, BEST_MODEL_DIR, NUM_EPOCHS, encoder),
            n_trials=trials_per_encoder)

        # Get the best trial for this encoder
        best_trial = study.best_trial
        best_trials[encoder] = best_trial.number
        print(f"Best trial for {encoder}: Trial {best_trial.number}, Value: {best_trial.value}, Params: {best_trial.params}")

        # Load the best model for this encoder
        trained_models[encoder] = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=len(CLASSES)).to(DEVICE)
        model_path = os.path.join(BEST_MODEL_DIR, f'best_model_{encoder}_trial_{best_trial.number}.pth')
        trained_models[encoder].load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Evaluation loop for each encoder
    CSV_FILENAME = os.path.join(BEST_MODEL_DIR, "trial_optuna.csv")
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Updated header to include per-class IoU scores
        header = ['Study Name', 'Encoder', 'Trial', 'Best Val Loss', 'Learning Rate', 'Weight Decay', 'Avg Test IoU', 'Avg Test Dice Loss'] + \
                 [f'Test_IoU_{cls}' for cls in CLASSES] + ['Best Epoch', 'IoU Plot Path']
        csv_writer.writerow(header)

        # Evaluate the best model for each encoder
        for encoder in encoders:
            if encoder not in trained_models:
                print(f"No trained model for {encoder}. Skipping evaluation.")
                continue

            # Update datasets with encoder-specific preprocessing
            preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
            train_dataset.dataset.datasets = [
                SegmentationDataset(ds.images_dir, ds.id_list, ds.output_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
                for ds in train_dataset.dataset.datasets]
            val_dataset.dataset.datasets = [
                SegmentationDataset(ds.images_dir, ds.id_list, ds.output_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
                for ds in val_dataset.dataset.datasets]
            test_dataset.datasets[0].preprocessing = get_preprocessing(preprocessing_fn)  # Update TestDataset
            test_dataset.datasets[1].dataset.datasets = [  # Update test_new_dataset
                SegmentationDataset(ds.images_dir, ds.id_list, ds.output_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn), verbose=False)
                for ds in test_dataset.datasets[1].dataset.datasets
            ]

            # Find the best trial for this encoder (already stored in best_trials)
            best_trial = studies[encoder].best_trial
            if best_trial is None:
                print(f"No completed trials for {encoder}. Skipping.")
                continue

            print(f"Evaluating best trial for {encoder}: Trial {best_trial.number}, Value: {best_trial.value}, Params: {best_trial.params}")

            # Load the best model (already loaded in trained_models)
            model = trained_models[encoder]
            model.eval()
            all_iou_scores = [[] for _ in range(len(CLASSES))]
            all_dice_losses = []
            with torch.no_grad():
                for images, masks in tqdm(test_dataloader, desc=f'Testing {encoder}'):
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    outputs = torch.sigmoid(model(images))
                    iou_scores = calculate_per_class_scores(outputs, masks)
                    loss_fn = CustomDiceLoss(num_classes=len(CLASSES), class_weights=torch.tensor(class_weights, device=DEVICE), device=DEVICE)
                    total_loss, dice_loss, ce_loss = loss_fn(outputs, masks, training=False)
                    all_dice_losses.append(dice_loss.cpu().item())
                    for i, score in enumerate(iou_scores):
                        all_iou_scores[i].append(score)

                    print(f"Test Batch ({encoder}) - Total Loss: {total_loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}")

            average_iou = [np.mean(scores) for scores in all_iou_scores]
            iou_scores_by_encoder[encoder] = average_iou
            test_dice_losses_by_encoder[encoder] = all_dice_losses
            avg_test_iou = np.mean(average_iou)
            avg_test_dice_loss = np.mean(all_dice_losses)
            print(f'Average IoU per class for {encoder}: {average_iou}')
            print(f'Average Dice Loss for {encoder}: {avg_test_dice_loss}')

            # Get the best epoch and IoU plot path from the trial's user attributes
            best_epoch = best_trial.user_attrs.get('best_epoch', -1)
            best_iou_plot_path = best_trial.user_attrs.get('best_iou_plot_path', 'N/A')

            # Log results with per-class IoU scores
            row = [f"{study_name_prefix}_{encoder}", encoder, best_trial.number, best_trial.value, best_trial.params['lr'], best_trial.params['weight_decay'], avg_test_iou, avg_test_dice_loss] + \
                  average_iou + [best_epoch, best_iou_plot_path]
            csv_writer.writerow(row)

            # Save only the best model for this encoder
            final_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_{encoder}.pth")
            torch.save(model.state_dict(), final_model_path)
            print(f"Saved best model for {encoder} at {final_model_path}")

    # Visualization using the combined function
    test_image_path = '/home/sonawane/backup/test_images/0°-Ansicht_Probe_1681.tif'
    test_image_dir = '/home/sonawane/backup/test_images/'
    
    plot_all_visualizations(encoders, trained_models, test_dice_losses_by_encoder, iou_scores_by_encoder, test_image_path, test_image_dir, CLASSES, DEVICE, BEST_MODEL_DIR, best_trials, studies)

if __name__ == "__main__":
    main()