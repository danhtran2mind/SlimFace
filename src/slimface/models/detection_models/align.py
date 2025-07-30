import torch
from PIL import Image
from typing import Union, List, Tuple
from . import mtcnn
from .face_yolo import face_yolo_detection

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize MTCNN model
MTCNN_MODEL = mtcnn.MTCNN(device=DEVICE, crop_size=(112, 112))

def add_image_padding(pil_img: Image.Image, top: int, right: int, bottom: int, left: int, 
                     color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Add padding to a PIL image."""
    width, height = pil_img.size
    new_width, new_height = width + right + left, height + top + bottom
    padded_img = Image.new(pil_img.mode, (new_width, new_height), color)
    padded_img.paste(pil_img, (left, top))
    return padded_img

def detect_faces_mtcnn(image: Union[str, Image.Image]) -> Tuple[Union[list, None], Union[Image.Image, None]]:
    """Detect and align faces using MTCNN model."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image or path to an image")
    
    try:
        bboxes, faces = MTCNN_MODEL.align_multi(image, limit=1)
        return bboxes[0] if bboxes else None, faces[0] if faces else None
    except Exception as e:
        print(f"MTCNN face detection failed: {e}")
        return None, None

def get_aligned_face(image_input: Union[str, List[str]], 
                    algorithm: str = 'mtcnn') -> List[Tuple[Union[list, None], Union[Image.Image, None]]]:
    """Get aligned faces from image(s) using specified algorithm."""
    if algorithm not in ['mtcnn', 'yolo']:
        raise ValueError("Algorithm must be 'mtcnn' or 'yolo'")

    # Convert single image path to list for consistent processing
    image_paths = [image_input] if isinstance(image_input, str) else image_input
    if not isinstance(image_paths, list):
        raise TypeError("Input must be a string or list of strings")

    if algorithm == 'mtcnn':
        return [detect_faces_mtcnn(path) for path in image_paths]
    
    # YOLO detection
    results = face_yolo_detection(
        image_paths,
        use_batch=True,
        device=DEVICE
    )
    return list(results)