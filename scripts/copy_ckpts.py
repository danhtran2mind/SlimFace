import shutil
import os

# Define source and destination directories
src_dir = "src/slim_face/models/edgeface/checkpoints"
dest_dir = "ckpts/edgeface_ckpts"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Copy all files and subdirectories from source to destination
for item in os.listdir(src_dir):
    src_path = os.path.join(src_dir, item)
    dest_path = os.path.join(dest_dir, item)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dest_path)

print(f"All files copied from {src_dir} to {dest_dir}")