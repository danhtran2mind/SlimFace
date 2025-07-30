import os
import sys
from PIL import Image

# Append the path to the inference script's directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'slimface', 'inference')))

from end2end_inference import inference_and_confirm

def run_inference(image, reference_dict_path, index_to_class_mapping_path, model_path,
                 edgeface_model_path="ckpts/idiap/edgeface_base.pt", 
                 algorithm="yolo", accelerator="auto", resolution=224, similarity_threshold=0.6):
    
    # Validate image input
    if image is None:
        return '<div class="error-message">Error: No image provided. Please upload an image.</div>'

    # Define temporary image path
    temp_image_path = os.path.join(os.path.dirname(__file__), "temp_data", "temp_image.jpg")
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    
    # Save the image
    try:
        image.save(temp_image_path)
    except Exception as e:
        return f'<div class="error-message">Error saving image: {str(e)}</div>'

    # Create args object to mimic command-line arguments
    class Args:
        def __init__(self):
            self.unknown_image_path = temp_image_path
            self.reference_dict_path = reference_dict_path.name if hasattr(reference_dict_path, 'name') else reference_dict_path
            self.index_to_class_mapping_path = index_to_class_mapping_path.name if hasattr(index_to_class_mapping_path, 'name') else index_to_class_mapping_path
            self.model_path = model_path.name if hasattr(model_path, 'name') else model_path
            self.edgeface_model_path = edgeface_model_path.name if hasattr(edgeface_model_path, 'name') else edgeface_model_path
            self.algorithm = algorithm
            self.accelerator = accelerator
            self.resolution = resolution
            self.similarity_threshold = similarity_threshold

    args = Args()

    # Validate inputs
    if not all([args.reference_dict_path, args.index_to_class_mapping_path, args.model_path]):
        return '<div class="error-message">Error: Please provide all required files (reference dict, index-to-class mapping, and model).</div>'

    try:
        # Call the inference function from end2end_inference.py
        results = inference_and_confirm(args)
        
        # Format output as HTML for Gradio
        output = '<div class="results-container">'
        output += '<h2 class="result-title">Inference Results</h2>'
        
        if not results:
            output += '<div class="error-message">No results returned from inference.</div>'
        else:
            for idx, result in enumerate(results, 1):
                output += '<div class="result-card">'
                output += f'<h3 class="result-title">Result {idx}</h3>'
                
                # Person Name
                person_name = result.get('predicted_class', 'N/A')
                output += f'<div class="result-item"><span class="label">Person Name</span><span class="value">{person_name}</span></div>'
                
                # Confidence
                confidence = result.get('confidence', 'N/A')
                confidence_str = f'{confidence:.4f}' if isinstance(confidence, (int, float)) else 'N/A'
                output += f'<div class="result-item"><span class="label">Confidence</span><span class="value">{confidence_str}</span></div>'
                
                # Similarity with Reference Image
                similarity = result.get('similarity', 'N/A')
                similarity_str = f'{similarity:.4f}' if isinstance(similarity, (int, float)) else 'N/A'
                output += f'<div class="result-item"><span class="label">Similarity with<br> Reference Image</span><span class="value">{similarity_str}</span></div>'
                
                # Confirmed Person
                confirmed = result.get('confirmed', 'N/A')
                confirmed_class = 'confirmed-true' if confirmed is True else 'confirmed-false' if confirmed is False else ''
                confirmed_str = str(confirmed) if confirmed is not None else 'N/A'
                output += f'<div class="result-item"><span class="label">Confirmed Person</span><span class="value {confirmed_class}">{confirmed_str}</span></div>'
                
                output += '</div>'
        
        output += '</div>'
        return output
    
    except Exception as e:
        return f'<div class="error-message">Error during inference: {str(e)}</div>'
    
    finally:
        # Clean up temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)