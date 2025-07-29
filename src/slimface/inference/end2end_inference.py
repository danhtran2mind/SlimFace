# from __future__ import annotations

# from pathlib import Path
# import cv2
# import gradio as gr
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from huggingface_hub import hf_hub_download

# # from utils import align_crop
# # from title import title_css, title_with_logo
# import torch.nn as nn
# import timm

#############
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "..", "..", "third_party"))
##########################################
import torch
from torchvision import transforms
from edgeface.face_alignment import align
from edgeface.backbones import get_model

# load model
model_name="edgeface_base"
model=get_model(model_name)
checkpoint_path=f'ckpts/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')) # Load state dict
model.eval() # Call eval() on the model object

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

paths = 'tests/test_images/Elon_Musk.jpg'
batch_size = len(paths) if isinstance(paths, (list, tuple)) else 1

# Align faces (assuming align.get_aligned_face returns a list of tuples)
aligned_result = align.get_aligned_face(paths, algorithm='yolo')

transformed_inputs = [transform(result[1]) for result in aligned_result]
transformed_inputs = torch.stack(transformed_inputs)

# Extract embeddings
embeddings = model(transformed_inputs)
print(embeddings.shape)  # Expected: torch.Size([batch_size, 512])
