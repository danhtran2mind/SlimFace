import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import warnings
import yaml
from torch import nn

# Append the parent directory's 'models/edgeface' folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from classification_models.alls import FaceClassifier
from detection_models import align

def resolve_path(path):
    """Convert a string like 'module.submodule.function' to a Python callable object."""
    try:
        module_name, obj_name = path.rsplit('.', 1)
        module = __import__("torchvision." + module_name, fromlist=[obj_name])
        return getattr(module, obj_name)
    except Exception as e:
        raise ValueError(f"Failed to resolve path {path}: {e}")

def load_model_configs(yaml_path):
    """Load model configurations from YAML file."""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        if 'models' in config:
            config = config['models']
        model_configs = {}
        for model_name, params in config.items():
            model_configs[model_name] = {
                'resolution': params['resolution'],
                'model_fn': resolve_path(params['model_fn']),
                'weights': params['weights'].split(".")[-1]
            }
        return model_configs
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {yaml_path} not found.")
    except Exception as e:
        raise ValueError(f"Error loading YAML configuration: {e}")

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

def load_model(model_path, classifier_head_path, model_name, num_classes, model_configs):
    """Load the trained model and classifier head."""
    model_fn = model_configs[model_name]['model_fn']
    weights = model_configs[model_name]['weights']
    base_model = model_fn(weights=weights)
    
    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    if hasattr(base_model, 'classifier'):
        base_model.classifier = nn.Identity()
    elif hasattr(base_model, 'fc'):
        base_model.fc = nn.Identity()
    elif hasattr(base_model, 'head'):
        base_model.head = nn.Identity()
    base_model.eval()
    
    model = FaceClassifier(base_model=base_model, num_classes=num_classes, model_name=model_name, model_configs=model_configs)
    
    # Load state dictionaries
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_class_mapping(dataset_dir):
    """Load class-to-index mapping from the dataset directory."""
    class_to_idx = {}
    for idx, person in enumerate(sorted(os.listdir(dataset_dir))):
        person_path = os.path.join(dataset_dir, person)
        if os.path.isdir(person_path):
            class_to_idx[person] = idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

def main(args):
    # Load model configurations
    MODEL_CONFIGS = load_model_configs(args.image_classification_models_config_path)
    if args.classification_model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {args.classification_model_name} not supported. Choose from {list(MODEL_CONFIGS.keys())}")
    
    resolution = MODEL_CONFIGS[args.classification_model_name]['resolution']
    
    # Load class mapping
    train_cache_dir = os.path.join(args.dataset_dir, f"train_data_aligned_{args.classification_model_name}")
    idx_to_class = load_class_mapping(train_cache_dir)
    num_classes = len(idx_to_class)
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        classifier_head_path=args.classifier_head_path,
        model_name=args.classification_model_name,
        num_classes=num_classes,
        model_configs=MODEL_CONFIGS
    )
    
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
            image_tensor = preprocess_image(image_path, algorithm=args.algorithm, resolution=resolution)
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = idx_to_class[predicted.item()]
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
    parser.add_argument('--dataset_dir', type=str, default='./data/processed_ds',
                        help='Path to the dataset directory to load class mappings.')
    parser.add_argument('--image_classification_models_config_path', type=str, 
                        default='./configs/image_classification_models_config.yaml',
                        help='Path to the YAML configuration file for model configurations.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained full model (.pth file).')
    parser.add_argument('--classifier_head_path', type=str,
                        help='Path to the trained classifier head (.pth file), if separate.')
    parser.add_argument('--classification_model_name', type=str, default='efficientnet_b0',
                        help='Model used for training.')
    parser.add_argument('--algorithm', type=str, default='yolo',
                        choices=['mtcnn', 'yolo'],
                        help='Face detection algorithm to use (mtcnn or yolo).')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['cpu', 'gpu', 'auto'],
                        help='Accelerator type for inference.')
    
    args = parser.parse_args()
    main(args)