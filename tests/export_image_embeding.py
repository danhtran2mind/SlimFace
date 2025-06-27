import os
import torch
from torchvision import transforms

# Import edgeface modules using relative paths
from edgeface.face_alignment import align
from edgeface.backbones import get_model

if __name__ == "__main__":
    batch_size = 1
    # load model
    model_name = "edgeface_s_gamma_05.pt" # or edgeface_xs_gamma_06
    model=get_model(model_name)
    checkpoint_path = os.path.join("edgeface", "checkpoints", model_name)

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')) # Load state dict
    
    model.eval() # Call eval() on the model object

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

    path = 'tests/test_images/Elon_Musk.jpg'
    aligned = align.get_aligned_face(path) # align face
    transformed_input = transform(aligned) # preprocessing
    transformed_input = transformed_input.reshape(batch_size, *transformed_input.shape)
    # extract embedding
    embedding = model(transformed_input)
    print(embedding.shape)
