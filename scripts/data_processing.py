from PIL import Image
import numpy as np
import os
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
import random
import shutil

RANDOM_RATIO = 0.5

def process_image(src_path, dest_dir, aug=None):
    """
    Process an image by resizing, normalizing, and optionally augmenting it.
    Saves both raw and augmented versions of the image.

    Args:
        src_path (str): Path to the source image
        dest_dir (str): Destination directory for the raw and augmented images
        aug (iaa.Sequential, optional): Augmentation pipeline
    Returns:
        list: List of saved image filenames (raw and optionally augmented)
    """
    saved_images = []
    try:
        # Open and process image
        img = Image.open(src_path).convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Save raw processed image
        raw_filename = os.path.basename(src_path)
        raw_dest_path = os.path.join(dest_dir, raw_filename)
        raw_img = Image.fromarray((img_array * 255).astype(np.uint8))
        raw_img.save(raw_dest_path, quality=100)
        saved_images.append(raw_filename)
        
        # Apply augmentation if specified and save augmented image
        if aug and random.random() <= RANDOM_RATIO:
            img_array_aug = aug.augment_image(img_array)
            # Clip values to ensure valid range after augmentation
            img_array_aug = np.clip(img_array_aug, 0, 1)
            # Convert back to image
            aug_img = Image.fromarray((img_array_aug * 255).astype(np.uint8))
            # Save augmented image with '_aug' suffix
            aug_filename = f"aug_{os.path.basename(src_path)}"
            aug_dest_path = os.path.join(dest_dir, aug_filename)
            aug_img.save(aug_dest_path, quality=100)
            saved_images.append(aug_filename)
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")
    
    return saved_images

def process_and_split_images(source_dir, person, temp_dir, train_person_dir, val_person_dir, test_split_rate, random_state, augment=False):
    """
    Process all images, optionally augment them, split into training and validation sets,
    and distribute to respective directories.

    Args:
        source_dir (str): Source directory containing images
        person (str): Person identifier (subdirectory name)
        temp_dir (str): Temporary directory to store processed images
        train_person_dir (str): Destination directory for training images
        val_person_dir (str): Destination directory for validation images
        test_split_rate (float): Proportion of data for validation set
        random_state (int): Random seed for reproducibility
        augment (bool): Whether to apply augmentation
    """
    # Define augmentation pipeline
    if augment:
        aug = iaa.Sequential([
            iaa.Fliplr(p=0.5),  # Horizontally flip images with 50% probability
            iaa.Affine(
                rotate=(-15, 15),  # Random rotation within ±15 degrees
            )
        ])
    else:
        aug = None

    # Ensure directories exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(val_person_dir, exist_ok=True)

    # Collect all images for the person
    images = [img for img in os.listdir(os.path.join(source_dir, person)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Initialize progress bar (if using tqdm)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(images), desc=f"Processing images for {person}")
    except ImportError:
        pbar = None  # Fallback if tqdm is not installed

    # Process and augment all images, saving to temporary directory
    all_image_filenames = []
    for img in images:
        src_path = os.path.join(source_dir, person, img)
        saved_images = process_image(src_path, temp_dir, aug)
        all_image_filenames.extend(saved_images)
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()

    # Split the combined list of raw and augmented images
    train_images, val_images = train_test_split(
        all_image_filenames, test_size=test_split_rate, random_state=random_state
    )

    # Move images to final training and validation directories
    for img in train_images:
        src = os.path.join(temp_dir, img)
        dst = os.path.join(train_person_dir, img)
        shutil.move(src, dst)
    for img in val_images:
        src = os.path.join(temp_dir, img)
        dst = os.path.join(val_person_dir, img)
        shutil.move(src, dst)

# Example usage
if __name__ == "__main__":
    source_dir = "path/to/source"
    person = "person_name"
    temp_dir = "path/to/temp"
    train_person_dir = "path/to/train/person_name"
    val_person_dir = "path/to/val/person_name"
    test_split_rate = 0.2
    random_state = 42
    augment = True

    process_and_split_images(
        source_dir=source_dir,
        person=person,
        temp_dir=temp_dir,
        train_person_dir=train_person_dir,
        val_person_dir=val_person_dir,
        test_split_rate=test_split_rate,
        random_state=random_state,
        augment=augment
    )
