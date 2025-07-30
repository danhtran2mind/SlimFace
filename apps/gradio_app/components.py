import os

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

CONTENT_DESCRIPTION = """
**SlimFace: Advanced Face Classification with TorchVision Backbones**
"""
CONTENT_IN = """
<style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        margin: 0; /* Remove default margin for full-width */
        padding: 20px; /* Adjust padding for content spacing */
        color: #333;
        width: 100%; /* Ensure body takes full width */
        box-sizing: border-box; /* Include padding in width calculation */
    }
    .attribution {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .quote-container {
        border-left: 5px solid #007bff;
        padding-left: 15px;
        margin-bottom: 15px;
        font-style: italic;
    }
    .attribution p {
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        border-radius: 4px;
        text-decoration: none;
        font-size: 14px;
        transition: background-color 0.3s;
    }
    .badge:hover {
        background-color: #0056b3;
    }
    .badge img {
        vertical-align: middle;
        margin-right: 5px;
    }
    .source {
        color: #555;
    }
</style>
    <div class="quote-container">
        <p>
            This project leverages code from 
            <a class="badge" href="https://github.com/otroshi/edgeface">
                <img src="https://img.shields.io/badge/Built%20on-otroshi%2Fedgeface-blue?style=flat&logo=github" alt="Built on edgeface">
            </a>
            by 
            <a class="badge" href="https://github.com/otroshi">
                <img src="https://img.shields.io/badge/GitHub-Hatef_Otroshi-blue?style=flat&logo=github" alt="Hatef Otroshi">
            </a>, 
            with our own bug fixes and enhancements available at 
            <a class="badge" href="https://github.com/danhtran2mind/edgeface/tree/main/face_alignment">
                <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2Fedgeface-blue?style=flat&logo=github" alt="Edgeface Enhancements">
            </a>.
        </p>
    </div>
    <p class="source">
        For more information, you can follow below:<br>
        Source code: 
        <a class="badge" href="https://github.com/danhtran2mind/SlimFace">
            <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2FSlimFace-blue?style=flat" alt="GitHub Repo">
            ,
        </a>
        Author: 
        <a class="badge" href="https://github.com/danhtran2mind">
            <img src="https://img.shields.io/badge/GitHub-danhtran2mind-blue?style=flat" alt="GitHub Profile">
            ,
        </a>
        PyTorch Docs: 
        <a class="badge" href="https://docs.pytorch.org/vision/main/models.html">
            <img src="https://img.shields.io/badge/PyTorch-Pretrain%20Model%20Docs-blue?style=flat" alt="PyTorch Docs">
        </a>
    </p>
"""

CONTENT_OUT = """
## More Information about SlimFace

SlimFace empowers developers to build high-accuracy face classification models using transfer learning, leveraging TorchVision's powerful pre-trained architectures. 🌟 It provides a flexible, efficient, and scalable solution for facial recognition, delivering top-tier performance for custom applications.

**Supported Architectures:**  
- **EfficientNet**: B0-B7 and V2 (Small, Medium, Large) for balanced performance and efficiency. 📸  
- **RegNet**: X/Y series (400MF to 128GF) for optimized computation across diverse hardware. 💻  
- **Vision Transformers (ViT)**: B_16, B_32, H_14, L_16, L_32 for cutting-edge feature extraction. 🚀  
"""