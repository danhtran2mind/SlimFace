from PIL import Image
import numpy as np
import os
import imgaug.augmenters as iaa

def process_image(src_path, dest_dir, aug=None):
    """
    Process an image by resizing, normalizing, and optionally augmenting it.
    Saves both raw and augmented versions of the image.

    Args:
        src_path (str): Path to the source image
        dest_dir (str): Destination directory for the raw processed image
        aug (iaa.Sequential, optional): Augmentation pipeline
    """
    try:
        # Open and process image
        img = Image.open(src_path).convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Save raw processed image
        raw_dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        raw_img = Image.fromarray((img_array * 255).astype(np.uint8))
        raw_img.save(raw_dest_path, quality=100)
        
        # Apply augmentation if specified and save augmented image
        if aug:
            # print("Augmentaion")
            img_array_aug = aug.augment_image(img_array)
            # Clip values to ensure valid range after augmentation
            img_array_aug = np.clip(img_array_aug, 0, 1)
            # Convert back to image
            aug_img = Image.fromarray((img_array_aug * 255).astype(np.uint8))
            # Save augmented image with '_aug' suffix
            aug_dest_path = os.path.join(dest_dir, f"aug_{os.path.basename(src_path)}")
            # print("Save to: ", aug_dest_path)
            aug_img.save(aug_dest_path, quality=100)
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")
