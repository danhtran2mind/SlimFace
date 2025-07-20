import os
import argparse
from huggingface_hub import snapshot_download


def download_models(model_name=None):
    """Download specified slimface models and index_to_class_mapping.json to ./ckpts.

    Args:
        model_name (str, optional): Specific model to download. If None, download all models.
    """
    repo_id = "danhtran2mind/slimface-sample-checkpoints"
    local_dir = "./ckpts"
    all_model_files = [
        "slimface_efficientnet_b3_full_model.pth",
        "slimface_efficientnet_v2_s_full_model.pth",
        "slimface_regnet_y_800mf_full_model.pth",
        "slimface_vit_b_16_full_model.pth",
        "index_to_class_mapping.json"
    ]

    # Determine files to download
    if model_name:
        if model_name not in all_model_files:
            raise ValueError(f"Model {model_name} not found in available models: {all_model_files[:-1]}")
        download_files = [model_name, "index_to_class_mapping.json"]
    else:
        download_files = all_model_files

    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=download_files,
            cache_dir=None,
            revision="main"
        )
        print(f"Downloaded {', '.join(download_files)} to {local_dir}")
    except Exception as e:
        print(f"Error downloading files: {e}")


def main():
    """Parse command-line arguments and initiate model download."""
    parser = argparse.ArgumentParser(description="Download slimface models from Hugging Face Hub.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "slimface_efficientnet_b3_full_model.pth",
            "slimface_efficientnet_v2_s_full_model.pth",
            "slimface_regnet_y_800mf_full_model.pth",
            "slimface_vit_b_16_full_model.pth"
        ],
        help="Specific model to download. If not provided, all models are downloaded."
    )
    args = parser.parse_args()

    download_models(args.model)


if __name__ == "__main__":
    main()