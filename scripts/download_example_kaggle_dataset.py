import os
import zipfile
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import process_image

def download_and_split_kaggle_dataset(dataset_slug, base_dir="data", img_size=(224, 224), augment=False):
    """
    Download a Kaggle dataset, split it into train/validation sets, and process images for face recognition.

    Args:
        dataset_slug (str): Dataset slug in 'username/dataset-name' format
        base_dir (str): Base directory for storing dataset
        img_size (tuple): Target image size (width, height) for resizing
        augment (bool): Whether to apply data augmentation
    """
    try:
        # Set up directories
        raw_dir = os.path.join(base_dir, "raw")
        processed_dir = os.path.join(base_dir, "processed_ds")
        zip_path = os.path.join(raw_dir, "dataset.zip")
        source_dir = os.path.join(raw_dir, "Faces", "Faces")
        train_dir = os.path.join(processed_dir, "train_data")
        val_dir = os.path.join(processed_dir, "val_data")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Download dataset with progress bar
        username, dataset_name = dataset_slug.split('/')
        if not (username and dataset_name):
            raise ValueError("Invalid dataset slug format. Expected 'username/dataset-name'")
        
        dataset_url = f"https://www.kaggle.com/api/v1/datasets/download/{username}/{dataset_name}"
        response = requests.get(dataset_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download dataset: {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading dataset",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

        # Organize and split dataset
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} not found")

        # Group files by person
        person_files = {}
        for file in os.listdir(source_dir):
            if os.path.isfile(os.path.join(source_dir, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                person = file.split("_")[0].replace(" ", "_")
                person_files.setdefault(person, []).append(file)

        # Define augmentation pipeline (if enabled)
        aug = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
            iaa.Affine(rotate=(-10, 10)),  # Random rotation
            iaa.GaussianBlur(sigma=(0, 0.5))  # Slight blur
        ]) if augment else None

        # Process and split files with progress bar
        total_files = sum(len(images) for images in person_files.values())
        with tqdm(total=total_files, desc="Processing and copying files", unit="file") as pbar:
            for person, images in person_files.items():
                train_person_dir = os.path.join(train_dir, person)
                val_person_dir = os.path.join(val_dir, person)
                os.makedirs(train_person_dir, exist_ok=True)
                os.makedirs(val_person_dir, exist_ok=True)

                train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

                for img in train_images:
                    process_image(os.path.join(source_dir, img), train_person_dir, img_size, aug)
                    pbar.update(1)
                for img in val_images:
                    process_image(os.path.join(source_dir, img), val_person_dir, img_size, None)  # No augmentation for validation
                    pbar.update(1)

        print(f"Dataset {dataset_slug} downloaded, extracted, processed, and split successfully!")

    except Exception as e:
        print(f"Error processing dataset: {e}")

# Example usage
if __name__ == "__main__":
    dataset_slug = "vasukipatel/face-recognition-dataset"
    download_and_split_kaggle_dataset(dataset_slug, img_size=(224, 224), augment=True)
