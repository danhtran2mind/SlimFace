import os
import zipfile
import requests
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import sys
import argparse
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import process_image

def download_and_split_kaggle_dataset(dataset_slug, base_dir="data", augment=False, random_state=42, test_split_rate=0.2, rotation_range=15, source_subdir="Original Images/Original Images"):
    """
    Download a Kaggle dataset, split it into train/validation sets, and process images for face recognition.

    Args:
        dataset_slug (str): Dataset slug in 'username/dataset-name' format.
        base_dir (str): Base directory for storing dataset.
        augment (bool): Whether to apply data augmentation to training images.
        random_state (int): Random seed for reproducibility in train-test split.
        test_split_rate (float): Proportion of data to use for validation (between 0 and 1).
        rotation_range (int): Maximum rotation angle in degrees for augmentation.
        source_subdir (str): Subdirectory within raw_dir containing images (default: 'Original Images/Original Images').
    """
    try:
        # Validate test_split_rate
        if not 0 < test_split_rate < 1:
            raise ValueError("test_split_rate must be between 0 and 1")

        # Set up directories
        raw_dir = os.path.join(base_dir, "raw")
        processed_dir = os.path.join(base_dir, "processed_ds")
        train_dir = os.path.join(processed_dir, "train_data")
        val_dir = os.path.join(processed_dir, "val_data")
        temp_dir = os.path.join(processed_dir, "temp")
        zip_path = os.path.join(raw_dir, "dataset.zip")
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Load Kaggle API credentials
        # kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        # if not os.path.exists(kaggle_json_path):
        #     raise FileNotFoundError("Kaggle API credentials not found. Please set up ~/.kaggle/kaggle.json.")
        # with open(kaggle_json_path, 'r') as f:
        #     kaggle_credentials = json.load(f)
        # username = kaggle_credentials.get('username')
        # key = kaggle_credentials.get('key')
        # if not (username and key):
        #     raise ValueError("Invalid Kaggle API credentials.")

        # Download dataset with progress bar
        username, dataset_name = dataset_slug.split('/')
        if not (username and dataset_name):
            raise ValueError("Invalid dataset slug format. Expected 'username/dataset-name'")
        
        dataset_url = f"https://www.kaggle.com/api/v1/datasets/download/{username}/{dataset_name}"
        # headers = {"Authorization": f"Bearer {key}"}
        print(f"Downloading dataset {dataset_slug}...")
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
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

        # Define source directory
        source_dir = os.path.join(raw_dir, source_subdir)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} not found")

        # Group files by person (subfolder names)
        person_files = {}
        for person in os.listdir(source_dir):
            person_dir = os.path.join(source_dir, person)
            if os.path.isdir(person_dir):
                person_files[person] = [
                    f for f in os.listdir(person_dir)
                    if os.path.isfile(os.path.join(person_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

        # Define augmentation pipeline
        if augment:
            aug = iaa.Sequential([
                iaa.Fliplr(p=1.0),  # Always flip images horizontally
                iaa.Sometimes(0.5,  # Apply the following with 50% probability
                    iaa.Affine(
                        rotate=(-rotation_range, rotation_range)  # Random rotation within ±degrees
                    )
                )
            ])
        else:
            aug = None

        # Process and split files with progress bar
        total_files = sum(len(images) for images in person_files.values())
        with tqdm(total=total_files, desc="Processing and copying files", unit="file") as pbar:
            for person, images in person_files.items():
                train_person_dir = os.path.join(train_dir, person)
                val_person_dir = os.path.join(val_dir, person)
                os.makedirs(train_person_dir, exist_ok=True)
                os.makedirs(val_person_dir, exist_ok=True)

                # Split images for this person
                train_images, val_images = train_test_split(
                    images, test_size=test_split_rate, random_state=random_state
                )

                # # Process images
                # all_image_filenames = []
                # for img in images:
                #     src_path = os.path.join(source_dir, person, img)
                #     saved_images = process_image(src_path, temp_dir, aug if img in train_images else None)
                #     all_image_filenames.extend(saved_images)
                #     pbar.update(1)

                # # Move images to final directories
                # for img in all_image_filenames:
                #     src = os.path.join(temp_dir, img)
                #     if not os.path.exists(src):
                #         print(f"Warning: File {src} not found, skipping.")
                #         continue
                #     if any(img in train_img or f"aug_{train_img}" == img for train_img in train_images):
                #         dst = os.path.join(train_person_dir, img)
                #     else:
                #         dst = os.path.join(val_person_dir, img)
                #     os.rename(src, dst)

                all_image_filenames = []
    
                # Ensure directories exist
                train_person_dir = os.path.join(train_dir, person)
                val_person_dir = os.path.join(val_dir, person)
                os.makedirs(train_person_dir, exist_ok=True)
                os.makedirs(val_person_dir, exist_ok=True)
                os.makedirs(temp_dir, exist_ok=True)
            
                # Process images
                for img in images:
                    src_path = os.path.join(source_dir, person, img)
                    # Append original image path
                    all_image_filenames.append(os.path.basename(src_path))
                    
                    # Process and save images (original and augmented) to temp directory
                    saved_images = process_image(src_path, temp_dir, aug if img in train_images else None)
                    all_image_filenames.extend(saved_images)
            
                # Split and move images to final train/val directories
                for img in all_image_filenames:
                    src = os.path.join(temp_dir, img)
                    if not os.path.exists(src):
                        print(f"Warning: File {src} not found, skipping.")
                        continue
                        
                    # Check if image is original or augmented training image
                    is_train = any(img == train_img or img.startswith(f"aug_{os.path.splitext(train_img)[0]}") 
                                  for train_img in train_images)
                    
                    # Move to appropriate directory
                    dst = os.path.join(train_person_dir if is_train else val_person_dir, img)
                    os.rename(src, dst)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"Dataset {dataset_slug} downloaded, extracted, processed, and split successfully!")

    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process a Kaggle dataset for face recognition.")
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default="vasukipatel/face-recognition-dataset",
        help="Kaggle dataset slug in 'username/dataset-name' format"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory for storing dataset"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for train-test split reproducibility"
    )
    parser.add_argument(
        "--test_split_rate",
        type=float,
        default=0.2,
        help="Proportion of data for validation (between 0 and 1)"
    )
    parser.add_argument(
        "--rotation_range",
        type=int,
        default=15,
        help="Maximum rotation angle in degrees for augmentation"
    )
    parser.add_argument(
        "--source_subdir",
        type=str,
        default="Original Images/Original Images",
        help="Subdirectory within raw_dir containing images"
    )

    args = parser.parse_args()

    download_and_split_kaggle_dataset(
        dataset_slug=args.dataset_slug,
        base_dir=args.base_dir,
        augment=args.augment,
        random_state=args.random_state,
        test_split_rate=args.test_split_rate,
        rotation_range=args.rotation_range,
        source_subdir=args.source_subdir
    )
