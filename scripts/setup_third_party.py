import os
import subprocess
import sys
import argparse

def run_command(command, error_message):
    """Run a shell command and handle errors."""
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"Error: {error_message}")
        sys.exit(1)

def setup_edgeface(repo_url, third_party_dir, branch=None):
    """Set up edgeface as a third-party dependency in the specified directory."""
    edgeface_dir = os.path.join(third_party_dir, "edgeface")

    # Create third_party directory if it doesn't exist
    if not os.path.exists(third_party_dir):
        os.makedirs(third_party_dir)
        print(f"Created directory: {third_party_dir}")

    # Clone edgeface if not already present
    if not os.path.exists(edgeface_dir):
        print(f"Cloning edgeface into {edgeface_dir}...")
        clone_command = f"git clone {repo_url} {edgeface_dir}"
        if branch:
            clone_command = f"git clone -b {branch} {repo_url} {edgeface_dir}"
        run_command(
            clone_command,
            f"Failed to clone edgeface from {repo_url}"
        )
    else:
        print(f"edgeface already exists at {edgeface_dir}")

    # Verify edgeface directory contains expected files
    if os.path.exists(edgeface_dir) and os.listdir(edgeface_dir):
        print(f"edgeface setup completed successfully at {edgeface_dir}")
    else:
        print(f"Error: edgeface directory is empty or invalid")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up edgeface as a third-party dependency.")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/danhtran2mind/edgeface.git",
        help="Git repository URL for edgeface (default: %(default)s)"
    )
    parser.add_argument(
        "--third-party-dir",
        default=os.path.join("src", "third_party"),
        help="Directory to store third-party dependencies (default: %(default)s)"
    )
    parser.add_argument(
        "--branch",
        help="Git branch to clone (optional)"
    )
    args = parser.parse_args()

    setup_edgeface(args.repo_url, args.third_party_dir, args.branch)