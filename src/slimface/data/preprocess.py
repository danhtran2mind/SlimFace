import os
import sys
from PIL import Image
from tqdm import tqdm
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.detection_models import align  # Assuming this is available in your project

def preprocess_and_cache_images(input_dir, output_dir, algorithm='yolo', resolution=224):
    """Preprocess images using face alignment and cache them with specified resolution."""
    if align is None:
        raise ImportError("face_alignment package is required for preprocessing.")
    os.makedirs(output_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
        for person in sorted(os.listdir(input_dir)):
            person_path = os.path.join(input_dir, person)
            if not os.path.isdir(person_path):
                continue
            output_person_path = os.path.join(output_dir, person)
            os.makedirs(output_person_path, exist_ok=True)
            skipped_count = 0
            for img_name in tqdm(os.listdir(person_path), desc=f"Processing {person}"):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(person_path, img_name)
                output_img_path = os.path.join(output_person_path, img_name)
                if os.path.exists(output_img_path):
                    skipped_count += 1
                    continue
                try:
                    aligned_result = align.get_aligned_face([img_path], algorithm=algorithm)
                    aligned_image = aligned_result[0][1] if aligned_result and len(aligned_result) > 0 else None
                    if aligned_image is None:
                        print(f"Face detection failed for {img_path}, using resized original image")
                        aligned_image = Image.open(img_path).convert('RGB')
                    aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    aligned_image = Image.open(img_path).convert('RGB')
                    aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
            if skipped_count > 0:
                print(f"Skipped {skipped_count} images for {person} that were already processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess and cache images with face alignment.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed images.')
    parser.add_argument('--algorithm', type=str, default='yolo', choices=['yolo', 'dlib'], help='Face detection algorithm to use.')
    parser.add_argument('--resolution', type=int, default=224, help='Resolution for the output images.')
    
    args = parser.parse_args()
    preprocess_and_cache_images(args.input_dir, args.output_dir, args.algorithm, args.resolution)

    # python src/slimface/data/preprocess.py \
    #     --input_dir "data/raw/Original Images/Original Images" \
    #     --output_dir "data/processed/Aligned Images" \
    #     --algorithm "yolo" \
    #     --resolution 224
