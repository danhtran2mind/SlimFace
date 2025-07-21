from __future__ import annotations

from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download

# from utils import align_crop
# from title import title_css, title_with_logo
import torch.nn as nn
import timm



##########################################
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import cv2

class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    """

    def __init__(self, model_name="edgenext_x_small", featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name

        self.model = timm.create_model(self.model_name)
        self.model.reset_classifier(self.featdim)

    def forward(self, x):
        x = self.model(x)
        return x


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "head" not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = False
            if module.bias is not None:
                bias = True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)


def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


import os
import torch
from huggingface_hub import hf_hub_download

model_configs = {
    "edgeface_base": {
        "repo": "idiap/EdgeFace-Base",
        "filename": "edgeface_base.pt",
        "timm_model": "edgenext_base",
        "post_setup": lambda x: x,
        "local_dir": "ckpts/idiap"
    },
    "edgeface_s_gamma_05": {
        "repo": "idiap/EdgeFace-S-GAMMA",
        "filename": "edgeface_s_gamma_05.pt",
        "timm_model": "edgenext_small",
        "post_setup": lambda x: replace_linear_with_lowrank_2(x, rank_ratio=0.5),
        "local_dir": "ckpts/idiap"
    },
    "edgeface_xs_gamma_06": {
        "repo": "idiap/EdgeFace-XS-GAMMA",
        "filename": "edgeface_xs_gamma_06.pt",
        "timm_model": "edgenext_x_small",
        "post_setup": lambda x: replace_linear_with_lowrank_2(x, rank_ratio=0.6),
        "local_dir": "ckpts/idiap"
    },
    "edgeface_xxs": {
        "repo": "idiap/EdgeFace-XXS",
        "filename": "edgeface_xxs.pt",
        "timm_model": "edgenext_xx_small",
        "post_setup": lambda x: x,
        "local_dir": "ckpts/idiap"
    },
}

def get_edge_model(name: str) -> torch.nn.Module:
    if not hasattr(get_edge_model, 'cache'):
        get_edge_model.cache = {}
    
    if name not in get_edge_model.cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = model_configs[name]
        # Define the local path using local_dir from config
        local_path = os.path.join(config["local_dir"], config["filename"])
        
        # Check if the model file exists in local_dir
        if os.path.exists(local_path):
            model_path = local_path
        else:
            # Download from Hugging Face to local_dir
            os.makedirs(config["local_dir"], exist_ok=True)
            model_path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["filename"],
                local_dir=config["local_dir"],
            )
        
        model = TimmFRWrapperV2(config["timm_model"], batchnorm=False)
        model = config["post_setup"](model)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.eval()
        model.to(device)
        get_edge_model.cache[name] = model
    return get_edge_model.cache[name]


get_edge_model.cache = {}

_tx = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

def compare(img_left, img_right, face_detection_algorithm, variant):
    crop_a = align.get_aligned_face([img_left], algorithm=face_detection_algorithm)
    crop_b = align.get_aligned_face([img_right], algorithm=face_detection_algorithm)
    
    if crop_a is None and crop_b is None:
        return None, None, "No face detected"
    if crop_a is None:
        return None, None, "No face in A"
    if crop_b is None:
        return None, None, "No face in B"
    mdl = get_edge_model(variant)
    dev = next(mdl.parameters()).device
    with torch.no_grad():
        ea = mdl(_tx(cv2.cvtColor(crop_a, cv2.COLOR_RGB2BGR))[None].to(dev))[0]
        eb = mdl(_tx(cv2.cvtColor(crop_b, cv2.COLOR_RGB2BGR))[None].to(dev))[0]
    pct = float(F.cosine_similarity(ea[None], eb[None]).item())
    pct = max(0, min(1, pct))
    colour = "#15803D" if pct >= 80 else "#CA8A04" if pct >= 50 else "#DC2626"
    return crop_a, crop_b, pct