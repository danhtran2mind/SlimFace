import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import warnings
import argparse

# Append necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edgeface.face_alignment import align
from edgeface.backbones import get_model
from models.detection_models import align as align_classifier

def preprocess_image(image_path, algorithm='yolo', resolution=224):
    """Preprocess a single image for classification."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
            aligned_result = align_classifier.get_aligned_face([image_path], algorithm=algorithm)
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
    return transform(aligned_image).unsqueeze(0)

def load_classifier_model(model_path):
    """Load the trained classification model in TorchScript format."""
    try:
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TorchScript model from {model_path}: {e}")

def load_class_mapping(index_to_class_mapping_path):
    """Load class-to-index mapping from the JSON file."""
    try:
        with open(index_to_class_mapping_path, 'r') as f:
            idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        return idx_to_class
    except Exception as e:
        raise ValueError(f"Error loading index to class mapping: {e}")

def get_edgeface_embeddings(image_path, model_name="edgeface_base", model_dir="ckpts/idiap"):
    """Extract embeddings from an image using the EdgeFace model."""
    model = get_model(model_name)
    checkpoint_path = f'{model_dir}/{model_name}.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    aligned_result = align.get_aligned_face(image_path, algorithm='yolo')
    if not aligned_result:
        raise ValueError(f"Face alignment failed for {image_path}")
    transformed_input = transform(aligned_result[0][1]).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(transformed_input)
    return embedding

def main(args):
    # Load class mapping
    idx_to_class = load_class_mapping(args.index_to_class_mapping_path)
    
    # Load classifier model
    classifier_model = load_classifier_model(args.model_path)
    
    # Process unknown image
    image_tensor = preprocess_image(args.unknown_image_path, algorithm=args.algorithm, resolution=args.resolution)
    with torch.no_grad():
        output = classifier_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = idx_to_class.get(predicted.item(), "Unknown")
    
    print(f"Predicted Class for {args.unknown_image_path}: {predicted_class}, Confidence: {confidence.item():.4f}")
    
    # Validate with embeddings if a reference image is provided for the predicted class
    reference_image_path = args.reference_images.get(predicted_class)
    if reference_image_path and os.path.exists(reference_image_path):
        print(f"Validating identity using EdgeFace embeddings against reference image for {predicted_class}...")
        unknown_embedding = get_edgeface_embeddings(args.unknown_image_path, args.edgeface_model_name, args.edgeface_model_dir)
        reference_embedding = get_edgeface_embeddings(reference_image_path, args.edgeface_model_name, args.edgeface_model_dir)
        
        similarity = torch.nn.functional.cosine_similarity(unknown_embedding, reference_embedding)
        print(f"Cosine similarity with {reference_image_path}: {similarity.item():.4f}")
        
        if similarity.item() >= args.similarity_threshold:
            print(f"Yes, the image {args.unknown_image_path} is confirmed to be {predicted_class}.")
        else:
            print(f"No, the image {args.unknown_image_path} is not confirmed to be {predicted_class} (similarity < {args.similarity_threshold}).")
    else:
        print(f"No reference image provided for {predicted_class} or reference image does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform face classification and validate identity using EdgeFace embeddings.')
    parser.add_argument('--unknown_image_path', type=str, required=True,
                        help='Path to the unknown image for classification.')
    parser.add_argument('--reference_images', type=str, required=True,
                        help='JSON string mapping class names to reference image paths (e.g., {"Elon_Musk": "path/to/elon.jpg"}).')
    parser.add_argument('--index_to_class_mapping_path', type=str, required=True,
                        help='Path to the JSON file containing index to class mapping.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained classifier model in TorchScript format (.pth file).')
    parser.add_argument('--edgeface_model_name', type=str, default='edgeface_base',
                        help='Name of the EdgeFace model to use.')
    parser.add_argument('--edgeface_model_dir', type=str, default='ckpts/idiap',
                        help='Directory where EdgeFace model checkpoints are stored.')
    parser.add_argument('--algorithm', type=str, default='yolo',
                        choices=['mtcnn', 'yolo'],
                        help='Face detection algorithm to use (mtcnn or yolo).')
    parser.add_argument('--resolution', type=int, default=224,
                        help='Resolution for input images (default: 224).')
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                        help='Cosine similarity threshold for identity confirmation (default: 0.6).')
    
    args = parser.parse_args()
    
    # Parse reference_images JSON string
    try:
        args.reference_images = json.loads(args.reference_images)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing reference_images JSON: {e}")
    
    main(args)