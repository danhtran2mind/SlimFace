import os
import argparse
from huggingface_hub import snapshot_download

# Model configurations for EdgeFace models
model_configs = {
    "edgeface_base": {
        "repo": "idiap/EdgeFace-Base",
        "filename": "edgeface_base.pt",
        "local_dir": "ckpts/idiap"
    },
    "edgeface_s_gamma_05": {
        "repo": "idiap/EdgeFace-S-GAMMA",
        "filename": "edgeface_s_gamma_05.pt",
        "local_dir": "ckpts/idiap"
    },
    "edgeface_xs_gamma_06": {
        "repo": "idiap/EdgeFace-XS-GAMMA",
        "filename": "edgeface_xs_gamma_06.pt",
        "local_dir": "ckpts/idiap"
    },
    "edgeface_xxs": {
        "repo": "idiap/EdgeFace-XXS",
        "filename": "edgeface_xxs.pt",
        "local_dir": "ckpts/idiap"
    },
    "SlimFace_efficientnet_b3": {
        "repo": "danhtran2mind/SlimFace-sample-checkpoints",
        "filename": "SlimFace_efficientnet_b3_full_model.pth",
        "local_dir": "ckpts"
    },
    "SlimFace_efficientnet_v2_s": {
        "repo": "danhtran2mind/SlimFace-sample-checkpoints",
        "filename": "SlimFace_efficientnet_v2_s_full_model.pth",
        "local_dir": "ckpts"
    },
    "SlimFace_regnet_y_800mf": {
        "repo": "danhtran2mind/SlimFace-sample-checkpoints",
        "filename": "SlimFace_regnet_y_800mf_full_model.pth",
        "local_dir": "ckpts"
    },
    "SlimFace_vit_b_16": {
        "repo": "danhtran2mind/SlimFace-sample-checkpoints",
        "filename": "SlimFace_vit_b_16_full_model.pth",
        "local_dir": "ckpts"
    },
    "SlimFace_mapping": {
        "repo": "danhtran2mind/SlimFace-sample-checkpoints",
        "filename": "index_to_class_mapping.json",
        "local_dir": "ckpts"
    }
}

def download_models(model_name=None):
    """Download specified models from model_configs to their respective local directories.

    Args:
        model_name (str, optional): Specific model to download. If None, download all models.
    """
    # Determine files to download
    if model_name:
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not found in available models: {list(model_configs.keys())}")
        configs_to_download = [model_configs[model_name]]
    else:
        configs_to_download = list(model_configs.values())

    for config in configs_to_download:
        repo_id = config["repo"]
        filename = config["filename"]
        local_dir = config["local_dir"]

        # Ensure the local directory exists
        os.makedirs(local_dir, exist_ok=True)

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=[filename],
                cache_dir=None,
                revision="main"
            )
            print(f"Downloaded {filename} to {local_dir}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

def main():
    """Parse command-line arguments and initiate model download."""
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(model_configs.keys()),
        help="Specific model to download. If not provided, all models are downloaded."
    )
    args = parser.parse_args()

    download_models(args.model)

if __name__ == "__main__":
    main()