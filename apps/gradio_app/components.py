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