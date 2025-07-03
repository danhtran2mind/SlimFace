from PIL import Image
import numpy as np
import os
import imgaug.augmenters as iaa
import random
import uuid

RANDOM_RATIO = 0.5 # 0.5
# TARGET_SIZE = (224, 224)  # Standard size for face recognition models

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
        
        # Resize image
        # img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Save raw processed image
        raw_filename = os.path.basename(src_path)
        base, ext = os.path.splitext(raw_filename)
        raw_dest_path = os.path.join(dest_dir, raw_filename)
        counter = 1
        while os.path.exists(raw_dest_path):
            raw_filename = f"{base}_{counter}{ext}"
            raw_dest_path = os.path.join(dest_dir, raw_filename)
            counter += 1
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
            # Save augmented image with unique suffix
            aug_filename = f"aug_{base}_{uuid.uuid4().hex[:8]}{ext}"
            aug_dest_path = os.path.join(dest_dir, aug_filename)
            aug_img.save(aug_dest_path, quality=100)
            saved_images.append(aug_filename)

    except Image.UnidentifiedImageError:
        print(f"Error: Cannot identify image file {src_path}")
    except OSError as e:
        print(f"Error processing image {src_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing image {src_path}: {e}")
    
    return saved_images
