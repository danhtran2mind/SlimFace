from PIL import Image
import numpy as np
import os
import imgaug.augmenters as iaa

def process_image(src_path, dest_dir, aug=None):
    """
    Process an image by resizing, normalizing, and optionally augmenting it.

    Args:
        src_path (str): Path to the source image
        dest_dir (str): Destination directory for the processed image
        aug (iaa.Sequential, optional): Augmentation pipeline
    """
    try:
        # Open and process image
        img = Image.open(src_path).convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Apply augmentation if specified
        if aug:
            img_array = aug.augment_image(img_array)
        
        # Clip values to ensure valid range after augmentation
        img_array = np.clip(img_array, 0, 1)
        
        # Convert back to image
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Save processed image
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        img.save(dest_path, quality=100)
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")
