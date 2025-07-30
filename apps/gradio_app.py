import os
import gradio as gr
from PIL import Image
from gradio_app.inference import run_inference

# File listing functions
def list_reference_files():
    ref_dir = "data/reference_data/"
    try:
        files = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".json")]
        return files if files else ["No .json files found in data/reference_data/"]
    except FileNotFoundError:
        return ["Directory data/reference_data/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_mapping_files():
    map_dir = "ckpts/"
    try:
        files = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".json")]
        return files if files else ["No .json files found in ckpts/"]
    except FileNotFoundError:
        return ["Directory ckpts/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_classifier_files():
    clf_dir = "ckpts/"
    try:
        files = [os.path.join(clf_dir, f) for f in os.listdir(clf_dir) if f.endswith(".pth")]
        return files if files else ["No .pth files found in ckpts/"]
    except FileNotFoundError:
        return ["Directory ckpts/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_edgeface_files():
    ef_dir = "ckpts/idiap/"
    try:
        files = [os.path.join(ef_dir, f) for f in os.listdir(ef_dir) if f.endswith(".pt")]
        return files if files else ["No .pt files found in ckpts/idiap/"]
    except FileNotFoundError:
        return ["Directory ckpts/idiap/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

# Custom CSS
custom_css = """
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(145deg, #e2e8f0 0%, #b8c6db 100%);
    margin: 0;
    padding: 0;
    min-height: 100vh;
    color: #1a202c;
}

.gradio-container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
    box-sizing: border-box;
}

h1 {
    color: #1a202c;
    font-size: 2.75rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 2.5rem;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #2b6cb0, #4a90e2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-title {
    color: #1a202c;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    border-bottom: 2px solid #4a90e2;
    padding-bottom: 0.5rem;
    letter-spacing: -0.015em;
}

.section-group {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 0.5rem;
    padding: 1.5rem;
    border: 1px solid rgba(226, 232, 240, 0.5);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.results-container {
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 1.25rem;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15), 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(226, 232, 240, 0.5);
    backdrop-filter: blur(8px);
}

.result-card {
    background: linear-gradient(145deg, #f7fafc, #edf2f7);
    border-radius: 1rem;
    padding: 2.25rem;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    position: relative;
    overflow: hidden;
}

.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.15);
    background: linear-gradient(145deg, #ffffff, #e6eefa);
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(to right, #4a90e2, #63b3ed);
    transition: height 0.3s ease;
}

.result-card:hover::before {
    height: 8px;
}

.result-title {
    color: #1a202c;
    font-size: 1.875rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #4a90e2;
    padding-bottom: 0.75rem;
    letter-spacing: -0.015em;
}

.result-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1rem 0;
    font-size: 1.125rem;
    color: #2d3748;
    line-height: 1.6;
}

.label {
    font-weight: 600;
    color: #2b6cb0;
    text-align: left;
    text-transform: uppercase;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    flex: 0 0 auto;
}

.value {
    color: #1a202c;
    font-weight: 500;
    text-align: right;
    flex: 0 0 auto;
}

.value.confirmed-true {
    color: #2f855a;
    font-weight: 600;
    background: #c6f6d5;
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
}

.value.confirmed-false {
    color: #c53030;
    font-weight: 600;
    background: #fed7d7;
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
}

.error-message {
    background: #fef2f2;
    color: #9b2c2c;
    padding: 1.75rem;
    border-radius: 0.875rem;
    margin: 1.25rem 0;
    font-size: 1.125rem;
    font-weight: 500;
    border: 1px solid #e53e3e;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.centered-button {
    display: block;
    margin: 1rem auto;
    background: #4a90e2;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.3s ease;
    position: relative;
    padding-left: 2.5rem;
    width: 30%;
}

.centered-button:hover {
    background: #2b6cb0;
}

.centered-button::after {
    content: '🤔';
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.2rem;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 1.5rem;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .results-container {
        padding: 1.5rem;
    }
    
    .result-card {
        padding: 1.5rem;
    }
    
    .result-title {
        font-size: 1.5rem;
    }
    
    .result-item {
        font-size: 1rem;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .label, .value {
        text-align: left;
    }
    
    .section-title {
        font-size: 1.25rem;
    }
    
    .section-group {
        padding: 1rem;
    }
    
    .centered-button {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
}
"""
CONTENT_MD = """
**SlimFace: Advanced Face Classification with TorchVision Backbones** 
*Source:* [GitHub Repo](https://img.shields.io/badge/GitHub-danhtran2mind%2FSlimFace-blue?style=flat)](https://github.com/danhtran2mind/SlimFace)  
*Author:* [![GitHub Profile](https://img.shields.io/badge/GitHub-danhtran2mind-blue?style=flat)](https://github.com/danhtran2mind)  
[![PyTorch Docs](https://img.shields.io/badge/PyTorch-Pretrain%20Model%20Docs-blue?style=flat)](https://docs.pytorch.org/vision/main/models.html)  

SlimFace is a cutting-edge project leveraging transfer learning to build high-performance face classification models from custom datasets. 🚀 Powered by TorchVision's pre-trained models, it integrates state-of-the-art architectures for robust and scalable facial recognition solutions.  

**Supported Architectures:**  
- **EfficientNet**: B0-B7, V2 (S, M, L) for optimized performance. 🖼️  
- **RegNet**: X/Y series (400MF to 128GF) for computational efficiency. ⚡  
- **Vision Transformers (ViT)**: B_16, B_32, H_14, L_16, L_32 for superior feature extraction. 🌟  

**Key Features from TorchVision:**  
- **Classification**: Access pre-trained weights (e.g., `ResNet50_Weights.DEFAULT`) via `torch.hub`. Preprocess with `weights.transforms()` for optimal results. 🧠  
- **Extensibility**: Supports tasks like semantic segmentation (FCN, DeepLabV3), object detection (Faster R-CNN), and video classification (R3D, Swin3D). 🎥  
- **Quantized Models**: INT8 support for efficient deployment (e.g., MobileNet_V3). 🔍  

Explore SlimFace for seamless integration of advanced face classification into real-world applications, and dive into [PyTorch Docs](https://docs.pytorch.org/vision/main/models.html) for more on TorchVision's capabilities! 🌐  
"""
# Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SlimFace Demonstration")
    gr.Markdown(CONTENT_MD)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            
        with gr.Column():
            output = gr.HTML(label="Inference Results", elem_classes=["results-container"])
    
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["section-group"]):
                gr.Markdown("### Model Files", elem_classes=["section-title"])
                ref_dict = gr.Dropdown(
                    choices=["Select a file"] + list_reference_files(),
                    label="Reference Dict JSON",
                    value="data/reference_data/reference_image_data.json"
                )
                index_map = gr.Dropdown(
                    choices=["Select a file"] + list_mapping_files(),
                    label="Index to Class Mapping JSON",
                    value="ckpts/index_to_class_mapping.json"
                )
                classifier_model = gr.Dropdown(
                    choices=["Select a file"] + list_classifier_files(),
                    label="Classifier Model (.pth)",
                    value="ckpts/SlimFace_efficientnet_b3_full_model.pth"
                )
                edgeface_model = gr.Dropdown(
                    choices=["Select a file"] + list_edgeface_files(),
                    label="EdgeFace Model (.pt)",
                    value="ckpts/idiap/edgeface_s_gamma_05.pt"
                )
        
        with gr.Column():
            with gr.Group(elem_classes=["section-group"]):
                gr.Markdown("### Advanced Settings", elem_classes=["section-title"])
                algorithm = gr.Dropdown(
                    choices=["yolo", "mtcnn", "retinaface"],
                    label="Detection Algorithm",
                    value="yolo"
                )
                accelerator = gr.Dropdown(
                    choices=["auto", "cpu", "cuda", "mps"],
                    label="Accelerator",
                    value="auto"
                )
                resolution = gr.Slider(
                    minimum=128,
                    maximum=512,
                    step=32,
                    label="Image Resolution",
                    value=300
                )
                similarity_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    label="Similarity Threshold",
                    value=0.3
                )
    
    with gr.Row():
        submit_btn = gr.Button("Run Inference", variant="primary",
                                elem_classes=["centered-button"])
    
    submit_btn.click(
        fn=run_inference,
        inputs=[
            image_input,
            ref_dict,
            index_map,
            classifier_model,
            edgeface_model,
            algorithm,
            accelerator,
            resolution,
            similarity_threshold
        ],
        outputs=output
    )
if __name__ == "__main__":
    demo.launch()