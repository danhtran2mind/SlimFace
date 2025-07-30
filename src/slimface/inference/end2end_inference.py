import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import warnings
import json

# Append necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edgeface.face_alignment import align as edgeface_align
from edgeface.backbones import get_model
from models.detection_models import align as align_classifier

def preprocess_image(image_path, algorithm='yolo', resolution=224):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
            aligned_result = align_classifier.get_aligned_face([image_path], algorithm=algorithm)
            aligned_image = aligned_result[0][1] if aligned_result and len(aligned_result) > 0 else Image.open(image_path).convert('RGB')
            aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        aligned_image = Image.open(image_path).convert('RGB').resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(aligned_image).unsqueeze(0)

def load_model(model_path):
    try:
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def load_class_mapping(index_to_class_mapping_path):
    try:
        with open(index_to_class_mapping_path, 'r') as f:
            idx_to_class = json.load(f)
        return {int(k): v for k, v in idx_to_class.items()}
    except Exception as e:
        raise ValueError(f"Error loading class mapping: {e}")

def get_edgeface_embeddings(image_path, model_path):
    """Get EdgeFace embeddings for a given image."""
    model_name = os.path.basename(model_path).split('.')[0]
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    aligned_result = edgeface_align.get_aligned_face(image_path, algorithm='yolo')
    if not aligned_result:
        raise ValueError(f"Face alignment failed for {image_path}")
    
    with torch.no_grad():
        return model(transform(aligned_result[0][1]).unsqueeze(0))

# def inference_and_confirm(args):
#     idx_to_class = load_class_mapping(args.index_to_class_mapping_path)
#     classifier_model = load_model(args.model_path)
#     device = torch.device('cuda' if torch.cuda.is_available() and args.accelerator == 'gpu' else 'cpu')
#     classifier_model = classifier_model.to(device)
    
#     # Load reference images mapping from JSON file
#     try:
#         with open(args.reference_dict_path, 'r') as f:
#             reference_images = json.load(f)
#     except Exception as e:
#         raise ValueError(f"Error loading reference images from {args.reference_dict_path}: {e}")
    
#     # Handle single image or directory
#     image_paths = [args.unknown_image_path] if args.unknown_image_path.endswith(('.jpg', '.jpeg', '.png')) else [
#         os.path.join(args.unknown_image_path, img) for img in os.listdir(args.unknown_image_path) 
#         if img.endswith(('.jpg', '.jpeg', '.png'))
#     ]
    
#     results = []
#     with torch.no_grad():
#         for image_path in image_paths:
#             image_tensor = preprocess_image(image_path, args.algorithm, args.resolution).to(device)
#             output = classifier_model(image_tensor)
#             probabilities = torch.softmax(output, dim=1)
#             confidence, predicted = torch.max(probabilities, 1)
#             predicted_class = idx_to_class.get(predicted.item(), "Unknown")
            
#             result = {'image_path': image_path, 'predicted_class': predicted_class, 'confidence': confidence.item()}
            
#             # Validate with EdgeFace embeddings if reference image exists
#             reference_image_path = reference_images.get(predicted_class)
#             if reference_image_path and os.path.exists(reference_image_path):
#                 unknown_embedding = get_edgeface_embeddings(image_path, args.edgeface_model_path)
#                 reference_embedding = get_edgeface_embeddings(reference_image_path, args.edgeface_model_path)
#                 similarity = torch.nn.functional.cosine_similarity(unknown_embedding, reference_embedding).item()
#                 result['similarity'] = similarity
#                 result['confirmed'] = similarity >= args.similarity_threshold
#             else:
#                 raise ValueError(f("Reference image for class '{predicted_class}' "
#                                    "not found in {args.reference_dict_path}"))
            
#             results.append(result)

#     #  {'image_path': 'tests/test_images/dont_know.jpg', 'predicted_class': 'Robert Downey Jr',
#     #  'confidence': 0.9292604923248291, 'similarity': 0.603316068649292, 'confirmed': True}

    return results

def inference_and_confirm(args):
    idx_to_class = load_class_mapping(args.index_to_class_mapping_path)
    classifier_model = load_model(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() and args.accelerator == 'gpu' else 'cpu')
    classifier_model = classifier_model.to(device)
    
    # Load reference images mapping from JSON file
    try:
        with open(args.reference_dict_path, 'r') as f:
            reference_images = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading reference images from {args.reference_dict_path}: {e}")
    
    # Handle single image or directory
    image_paths = [args.unknown_image_path] if args.unknown_image_path.endswith(('.jpg', '.jpeg', '.png')) else [
        os.path.join(args.unknown_image_path, img) for img in os.listdir(args.unknown_image_path) 
        if img.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    results = []
    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = preprocess_image(image_path, args.algorithm, args.resolution).to(device)
            output = classifier_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = idx_to_class.get(predicted.item(), "Unknown")
            
            result = {'image_path': image_path, 'predicted_class': predicted_class, 'confidence': confidence.item()}
            
            # Validate with EdgeFace embeddings if reference image exists
            reference_image_path = reference_images.get(predicted_class)
            if reference_image_path and os.path.exists(reference_image_path):
                unknown_embedding = get_edgeface_embeddings(image_path, args.edgeface_model_path)
                reference_embedding = get_edgeface_embeddings(reference_image_path, args.edgeface_model_path)
                similarity = torch.nn.functional.cosine_similarity(unknown_embedding, reference_embedding).item()
                result['similarity'] = similarity
                result['confirmed'] = similarity >= args.similarity_threshold
            else:
                result['similarity'] = None
                result['confirmed'] = False
            
            results.append(result)

    return results

def main(args):
    results = inference_and_confirm(args)
    for result in results:
        print(f"Image: {result['image_path']}, Predicted Class: {result['predicted_class']}, "
              f"Confidence: {result['confidence']:.4f}, Similarity: {result.get('similarity', 'N/A'):.4f}, "
              f"Confirmed: {result.get('confirmed', 'N/A')}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face classification with EdgeFace embedding validation.')
    parser.add_argument('--unknown_image_path', type=str, required=True, help='Path to image or directory.')
    parser.add_argument('--reference_dict_path', type=str, required=True, help='Path to JSON file mapping classes to reference image paths.')
    parser.add_argument('--index_to_class_mapping_path', type=str, required=True, help='Path to index-to-class JSON.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to classifier model (.pth).')
    parser.add_argument('--edgeface_model_path', type=str, default='ckpts/idiap/edgeface_base.pt', help='EdgeFace model path.')
    # parser.add_argument('--edgeface_model_dir', type=str, default='ckpts/idiap', help='EdgeFace model directory.')
    parser.add_argument('--algorithm', type=str, default='yolo', choices=['mtcnn', 'yolo'], help='Face detection algorithm.')
    parser.add_argument('--accelerator', type=str, default='auto', choices=['cpu', 'gpu', 'auto'], help='Accelerator type.')
    parser.add_argument('--resolution', type=int, default=224, help='Input image resolution.')
    parser.add_argument('--similarity_threshold', type=float, default=0.6, help='Cosine similarity threshold.')
    
    args = parser.parse_args()
    main(args)

    # python src/slimface/inference/end2end_inference.py \
    # --unknown_image_path tests/test_images/dont_know.jpg \
    # --reference_dict_path tests/reference_image_data.json \
    # --index_to_class_mapping_path /content/SlimFace/ckpts/index_to_class_mapping.json \
    # --model_path /content/SlimFace/ckpts/SlimFace_efficientnet_b3_full_model.pth \
    # --edgeface_model_name edgeface_base \
    # --similarity_threshold 0.6