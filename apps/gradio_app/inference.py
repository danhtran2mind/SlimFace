import os
import sys
from PIL import Image

# Append the path to the inference script's directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'slimface', 'inference')))
from end2end_inference import inference_and_confirm

def run_inference(image, reference_dict_path, index_to_class_mapping_path, model_path, 
                 edgeface_model_name="edgeface_base", edgeface_model_dir="ckpts/idiap", 
                 algorithm="yolo", accelerator="auto", resolution=224, similarity_threshold=0.6):
    # Save uploaded image temporarily in apps/gradio_app/
    temp_image_path = os.path.join(os.path.dirname(__file__), "temp_image.jpg")
    image.save(temp_image_path)

    # Create args object to mimic command-line arguments
    class Args:
        def __init__(self):
            self.unknown_image_path = temp_image_path
            self.reference_dict_path = reference_dict_path.name if reference_dict_path else None
            self.index_to_class_mapping_path = index_to_class_mapping_path.name if index_to_class_mapping_path else None
            self.model_path = model_path.name if model_path else None
            self.edgeface_model_name = edgeface_model_name
            self.edgeface_model_dir = edgeface_model_dir
            self.algorithm = algorithm
            self.accelerator = accelerator
            self.resolution = resolution
            self.similarity_threshold = similarity_threshold

    args = Args()

    # Validate inputs
    if not all([args.reference_dict_path, args.index_to_class_mapping_path, args.model_path]):
        return "Error: Please provide all required files (reference dict, index-to-class mapping, and model)."

    try:
        # Call the inference function from end2end_inference.py
        results = inference_and_confirm(args)
        
        # Format output
        output = ""
        for result in results:
            output += f"Image: {result['image_path']}\n"
            output += f"Predicted Class: {result['predicted_class']}\n"
            output += f"Confidence: {result['confidence']:.4f}\n"
            output += f"Similarity: {result.get('similarity', 'N/A'):.4f}\n"
            output += f"Confirmed: {result.get('confirmed', 'N/A')}\n\n"
        
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        # Clean up temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)