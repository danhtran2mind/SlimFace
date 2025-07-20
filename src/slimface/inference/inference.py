import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import warnings
import json


# Append the parent directory's 'models/edgeface' folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.detection_models import align

def preprocess_image(image_path, algorithm='yolo', resolution=224):
    """Preprocess a single image using face alignment and specified resolution."""
    if align is None:
        raise ImportError("face_alignment package is required for preprocessing.")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
            aligned_result = align.get_aligned_face([image_path], algorithm=algorithm)
            aligned_image = aligned_result[0][1] if aligned_result and len(aligned_result) > 0 else None
            if aligned_image is None:
                print(f"Face detection failed for {image_path}, using resized original image")
                aligned_image = Image.open(image_path).convert('RGB')
            aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        aligned_image = Image.open(image_path).convert('RGB')
        aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(aligned_image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def load_model(model_path):
    """Load the trained model in TorchScript format."""
    try:
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TorchScript model from {model_path}: {e}")

def load_class_mapping(idx_to_class_path):
    """Load class-to-index mapping from the JSON file."""
    try:
        with open(idx_to_class_path, 'r') as f:
            idx_to_class = json.load(f)
        # Convert string keys (from JSON) to integers
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        return idx_to_class
    except FileNotFoundError:
        raise FileNotFoundError(f"Index to class mapping file {idx_to_class_path} not found.")
    except Exception as e:
        raise ValueError(f"Error loading index to class mapping: {e}")

def main(args):
    # Load class mapping from JSON file
    idx_to_class = load_class_mapping(args.idx_to_class_path)
    
    # Load model
    model = load_model(args.model_path)
    
    # Process input images
    device = torch.device('cuda' if torch.cuda.is_available() and args.accelerator == 'gpu' else 'cpu')
    model = model.to(device)
    
    image_paths = []
    if os.path.isdir(args.input_path):
        for img_name in os.listdir(args.input_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.input_path, img_name))
    else:
        if args.input_path.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(args.input_path)
        else:
            raise ValueError("Input path must be a directory or a valid image file.")
    
    # Perform inference
    results = []
    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = preprocess_image(image_path, algorithm=args.algorithm, resolution=args.resolution)
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = idx_to_class.get(predicted.item(), "Unknown")
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence.item()
            })
    
    # Output results
    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference with a trained face classification model.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to an image or directory of images for inference.')
    parser.add_argument('--index_to_class_mapping_path', type=str, required=True,
                        help='Path to the JSON file containing index to class mapping.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained full model in TorchScript format (.pth file).')
    parser.add_argument('--algorithm', type=str, default='yolo',
                        choices=['mtcnn', 'yolo'],
                        help='Face detection algorithm to use (mtcnn or yolo).')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['cpu', 'gpu', 'auto'],
                        help='Accelerator type for inference.')
    parser.add_argument('--resolution', type=int, default=224,
                        help='Resolution for input images (default: 224).')
    
    args = parser.parse_args()
    main(args)