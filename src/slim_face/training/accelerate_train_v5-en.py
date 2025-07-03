import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights
)

# Append the parent directory's 'models/edgeface' folder to the system path to allow importing modules from that location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'edgeface')))
try:
    from face_alignment import align
    from backbones import get_model    
except ImportError:
    print("Warning: face_alignment package not found. Ensure it is installed for preprocessing.")
    align = None

# Mapping of EfficientNet versions to their input resolutions
EFFICIENTNET_RESOLUTIONS = {
    'b0': 224,
    'b1': 240,
    'b2': 260,
    'b3': 300,
    'b4': 380,
    'b5': 456,
    'b6': 528,
    'b7': 600
}

# Mapping of EfficientNet versions to their model and weights
EFFICIENTNET_MODELS = {
    'b0': (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    'b1': (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    'b2': (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    'b3': (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    'b4': (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    'b5': (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
    'b6': (efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1),
    'b7': (efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1)
}

def preprocess_and_cache_images(input_dir, output_dir, algorithm='yolo', resolution=224):
    """Preprocess images using face alignment and cache them with specified resolution."""
    if align is None:
        raise ImportError("face_alignment package is required for preprocessing.")
    os.makedirs(output_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
        for person in sorted(os.listdir(input_dir)):
            person_path = os.path.join(input_dir, person)
            if not os.path.isdir(person_path):
                continue
            output_person_path = os.path.join(output_dir, person)
            os.makedirs(output_person_path, exist_ok=True)
            skipped_count = 0
            for img_name in tqdm(os.listdir(person_path), desc=f"Processing {person}"):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(person_path, img_name)
                output_img_path = os.path.join(output_person_path, img_name)
                if os.path.exists(output_img_path):
                    skipped_count += 1
                    continue
                try:
                    aligned_result = align.get_aligned_face([img_path], algorithm=algorithm)
                    aligned_image = aligned_result[0][1] if aligned_result and len(aligned_result) > 0 else None
                    if aligned_image is None:
                        print(f"Face detection failed for {img_path}, using resized original image")
                        aligned_image = Image.open(img_path).convert('RGB')
                    aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    aligned_image = Image.open(img_path).convert('RGB')
                    aligned_image = aligned_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
            if skipped_count > 0:
                print(f"Skipped {skipped_count} images for {person} that were already processed.")

class FaceDataset(Dataset):
    """Dataset for loading pre-aligned face images."""
    def __init__(self, root_dir, transform=None, resolution=224):
        self.root_dir = root_dir
        self.transform = transform
        self.resolution = resolution
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        for idx, person in enumerate(sorted(os.listdir(root_dir))):
            person_path = os.path.join(root_dir, person)
            if os.path.isdir(person_path):
                self.class_to_idx[person] = idx
                for img_name in os.listdir(person_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(person_path, img_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # Ensure image is resized to the correct resolution
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.resolution, self.resolution))
        if self.transform:
            image = self.transform(image)
        return image, label

class FaceClassifier(nn.Module):
    """Face classification model with a convolutional head."""
    def __init__(self, base_model, num_classes):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        # Determine the number of input channels for the conv head based on the base model
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Temporary input to get feature size
            features = base_model.features(dummy_input)
            in_channels = features.shape[1]
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract feature maps from EfficientNet
        features = self.base_model.features(x)
        output = self.conv_head(features)
        return output

class FaceClassifierLightning(pl.LightningModule):
    """PyTorch Lightning module for face classification."""
    def __init__(self, base_model, num_classes, learning_rate, warmup_steps=1000, total_steps=100000, max_lr_factor=10.0):
        super(FaceClassifierLightning, self).__init__()
        self.model = FaceClassifier(base_model, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = learning_rate * max_lr_factor
        self.min_lr = 1e-6
        self.save_hyperparameters("num_classes", "learning_rate", "warmup_steps", "total_steps", "max_lr_factor")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics
        train_loss = metrics.get('train_loss_epoch', 0.0)
        train_acc = metrics.get('train_acc_epoch', 0.0)
        val_loss = metrics.get('val_loss_epoch', 0.0)
        val_acc = metrics.get('val_acc_epoch', 0.0)
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f"\nEpoch {self.current_epoch + 1}: "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, "
              f"Learning rate: {current_lr:.6e}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.conv_head.parameters(), lr=self.learning_rate)
        def lr_lambda(step):
            if step < self.warmup_steps:
                return (self.max_lr - self.learning_rate) / self.warmup_steps * step + self.learning_rate
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_lr = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_lr
            return max(lr, self.min_lr) / self.learning_rate
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

class CustomModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, metrics, ver=None):
        metrics['epoch'] = metrics.get('epoch', 0) + 1
        return super().format_checkpoint_name(metrics, ver)

class CustomTQDMProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["epoch"] = trainer.current_epoch + 1
        return items
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description(f"Training Epoch {self.trainer.current_epoch + 1}")
        return bar
     def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        # Ensure total is set to avoid None issues during sanity check
        if self.trainer.sanity_checking:
            bar.total = self.trainer.num_sanity_val_steps or 1
        bar.set_description(f"{'Sanity Checking' if self.trainer.sanity_checking else 'Validating'} Epoch {self.trainer.current_epoch + 1}")
        return bar    
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_progress_bar:
            self.train_progress_bar.set_description(f"Training Epoch {trainer.current_epoch + 1}")

def main(args):
    mp.set_start_method('spawn', force=True)
    
    # Get the resolution for the selected EfficientNet version
    resolution = EFFICIENTNET_RESOLUTIONS.get(args.efficientnet_version, 224)
    
    train_cache_dir = os.path.join(args.dataset_dir, f"train_data_aligned_{args.efficientnet_version}")
    val_cache_dir = os.path.join(args.dataset_dir, f"val_data_aligned_{args.efficientnet_version}")
    print(f"Preprocessing training dataset with resolution {resolution}...")
    preprocess_and_cache_images(
        input_dir=os.path.join(args.dataset_dir, "train_data"),
        output_dir=train_cache_dir,
        algorithm=args.algorithm,
        resolution=resolution
    )
    print(f"Preprocessing validation dataset with resolution {resolution}...")
    preprocess_and_cache_images(
        input_dir=os.path.join(args.dataset_dir, "val_data"),
        output_dir=val_cache_dir,
        algorithm=args.algorithm,
        resolution=resolution
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = FaceDataset(root_dir=train_cache_dir, transform=transform, resolution=resolution)
    val_dataset = FaceDataset(root_dir=val_cache_dir, transform=transform, resolution=resolution)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is empty. Check dataset directory or preprocessing.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train DataLoader is empty. Check dataset size or batch configuration.")
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_steps * total_steps) if args.warmup_steps > 0 else int(0.05 * total_steps)
    
    # Load the appropriate EfficientNet model
    model_fn, weights = EFFICIENTNET_MODELS.get(args.efficientnet_version, (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1))
    base_model = model_fn(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.classifier = nn.Identity()  # Remove classifier
    base_model.eval()
    
    model = FaceClassifierLightning(
        base_model=base_model,
        num_classes=len(train_dataset.class_to_idx),
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr_factor=args.max_lr_factor
    )
    checkpoint_callback = CustomModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename=f'face_classifier_{args.efficientnet_version}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min'
    )
    
    progress_bar = CustomTQDMProgressBar()
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a face classification model with PyTorch Lightning.')
    parser.add_argument('--dataset_dir', type=str, default='./data/processed_ds',
                        help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Initial learning rate for the optimizer.')
    parser.add_argument('--max_lr_factor', type=float, default=10.0,
                        help='Factor to multiply initial learning rate to get max learning rate during warmup.')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['cpu', 'gpu', 'tpu', 'auto'],
                        help='Accelerator type for training.')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use (e.g., number of GPUs).')
    parser.add_argument('--algorithm', type=str, default='yolo',
                        choices=['mtcnn', 'yolo'],
                        help='Face detection algorithm to use (mtcnn or yolo).')
    parser.add_argument('--warmup_steps', type=float, default=0.05,
                        help='Fraction of total steps for warmup phase (e.g., 0.05 for 5%).')
    parser.add_argument('--total_steps', type=int, default=0,
                        help='Total number of training steps (0 to use epochs * steps_per_epoch).')
    parser.add_argument('--efficientnet_version', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                        help='EfficientNet version to use (b0 to b7).')

    args = parser.parse_args()
    main(args)
