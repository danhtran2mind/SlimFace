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

CONTENT_IN_1 = """
SlimFace empowers developers to build high-accuracy face classification models by leveraging TorchVision's powerful pre-trained architectures through a transfer learning approach. 🌟 It provides a flexible, efficient, and scalable solution for facial recognition, supporting training on custom datasets and inference to deliver top-tier performance for custom applications.
"""
CONTENT_IN_2 = """
<p class="source">
    For more information, you can follow below:<br>
    Source code: 
    <a class="badge" href="https://github.com/danhtran2mind/SlimFace">
        <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2FSlimFace-blue?style=flat" alt="GitHub Repo">
    </a>,
    Author: 
    <a class="badge" href="https://github.com/danhtran2mind">
        <img src="https://img.shields.io/badge/GitHub-danhtran2mind-blue?style=flat" alt="GitHub Profile">
    </a>,
    PyTorch Docs: 
    <a class="badge" href="https://docs.pytorch.org/vision/main/models.html">
        <img src="https://img.shields.io/badge/PyTorch-Pretrain%20Model%20Docs-blue?style=flat" alt="PyTorch Docs">
    </a>
</p>
"""


CONTENT_OUTTRO = """
## More Information about SlimFace
"""
CONTENT_OUT_1 = """
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
"""
CONTENT_OUT_2 = """
**Supported Architectures:**  
- **EfficientNet**: B0-B7 and V2 (Small, Medium, Large) for balanced performance and efficiency. 📸  
- **RegNet**: X/Y series (400MF to 128GF) for optimized computation across diverse hardware. 💻  
- **Vision Transformers (ViT)**: B_16, B_32, H_14, L_16, L_32 for cutting-edge feature extraction. 🚀  
"""