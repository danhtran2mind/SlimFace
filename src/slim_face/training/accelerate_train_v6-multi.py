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
import torchvision.models as models

# Append the parent directory's 'models/edgeface' folder to the system path to allow importing modules from that location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'edgeface')))
try:
    from face_alignment import align
    from backbones import get_model    
except ImportError:
    print("Warning: face_alignment package not found. Ensure it is installed for preprocessing.")
    align = None

# Mapping of model names to their input resolutions and weights
MODEL_CONFIGS = {
    'efficientnet_b0': {'resolution': 224, 'model_fn': models.efficientnet_b0, 'weights': models.EfficientNet_B0_Weights.IMAGENET1K_V1},
    'efficientnet_b1': {'resolution': 240, 'model_fn': models.efficientnet_b1, 'weights': models.EfficientNet_B1_Weights.IMAGENET1K_V1},
    'efficientnet_b2': {'resolution': 260, 'model_fn': models.efficientnet_b2, 'weights': models.EfficientNet_B2_Weights.IMAGENET1K_V1},
    'efficientnet_b3': {'resolution': 300, 'model_fn': models.efficientnet_b3, 'weights': models.EfficientNet_B3_Weights.IMAGENET1K_V1},
    'efficientnet_b4': {'resolution': 380, 'model_fn': models.efficientnet_b4, 'weights': models.EfficientNet_B4_Weights.IMAGENET1K_V1},
    'efficientnet_b5': {'resolution': 456, 'model_fn': models.efficientnet_b5, 'weights': models.EfficientNet_B5_Weights.IMAGENET1K_V1},
    'efficientnet_b6': {'resolution': 528, 'model_fn': models.efficientnet_b6, 'weights': models.EfficientNet_B6_Weights.IMAGENET1K_V1},
    'efficientnet_b7': {'resolution': 600, 'model_fn': models.efficientnet_b7, 'weights': models.EfficientNet_B7_Weights.IMAGENET1K_V1},
    'efficientnet_v2_s': {'resolution': 384, 'model_fn': models.efficientnet_v2_s, 'weights': models.EfficientNet_V2_S_Weights.IMAGENET1K_V1},
    'efficientnet_v2_m': {'resolution': 480, 'model_fn': models.efficientnet_v2_m, 'weights': models.EfficientNet_V2_M_Weights.IMAGENET1K_V1},
    'efficientnet_v2_l': {'resolution': 480, 'model_fn': models.efficientnet_v2_l, 'weights': models.EfficientNet_V2_L_Weights.IMAGENET1K_V1},
    'regnet_y_400mf': {'resolution': 224, 'model_fn': models.regnet_y_400mf, 'weights': models.RegNet_Y_400MF_Weights.IMAGENET1K_V2},
    'regnet_y_800mf': {'resolution': 224, 'model_fn': models.regnet_y_800mf, 'weights': models.RegNet_Y_800MF_Weights.IMAGENET1K_V2},
    'regnet_y_1_6gf': {'resolution': 224, 'model_fn': models.regnet_y_1_6gf, 'weights': models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2},
    'regnet_y_3_2gf': {'resolution': 224, 'model_fn': models.regnet_y_3_2gf, 'weights': models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2},
    'regnet_y_8gf': {'resolution': 224, 'model_fn': models.regnet_y_8gf, 'weights': models.RegNet_Y_8GF_Weights.IMAGENET1K_V2},
    'regnet_y_16gf': {'resolution': 224, 'model_fn': models.regnet_y_16gf, 'weights': models.RegNet_Y_16GF_Weights.IMAGENET1K_V2},
    'regnet_y_32gf': {'resolution': 224, 'model_fn': models.regnet_y_32gf, 'weights': models.RegNet_Y_32GF_Weights.IMAGENET1K_V2},
    'regnet_y_128gf': {'resolution': 224, 'model_fn': models.regnet_y_128gf, 'weights': models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1},
    'vit_b_16': {'resolution': 224, 'model_fn': models.vit_b_16, 'weights': models.ViT_B_16_Weights.IMAGENET1K_V1},
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
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.resolution, self.resolution))
        if self.transform:
            image = self.transform(image)
        return image, label

# class FaceClassifier(nn.Module):
#     """Face classification model with a convolutional head."""
#     def __init__(self, base_model, num_classes, model_name):
#         super(FaceClassifier, self).__init__()
#         self.base_model = base_model
#         self.model_name = model_name
        
#         # Determine the feature extraction method based on model type
#         if 'efficientnet' in model_name:
#             with torch.no_grad():
#                 dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
#                 features = base_model.features(dummy_input)
#                 in_channels = features.shape[1]
#             self.feature_dim = in_channels
#         elif 'regnet' in model_name:
#             with torch.no_grad():
#                 dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
#                 features = base_model.features(dummy_input) if hasattr(base_model, 'features') else base_model(dummy_input)
#                 in_channels = features.shape[1]
#             self.feature_dim = in_channels
#         elif 'vit' in model_name:
#             with torch.no_grad():
#                 dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
#                 features = base_model(dummy_input)
#                 in_channels = features.shape[-1]
#             self.feature_dim = in_channels
#         else:
#             raise ValueError(f"Unsupported model type: {model_name}")

#         # Define the classifier head
#         if 'vit' in model_name:
#             self.conv_head = nn.Sequential(
#                 nn.Linear(self.feature_dim, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(),
#                 nn.Linear(256, num_classes)
#             )
#         else:
#             self.conv_head = nn.Sequential(
#                 nn.Conv2d(self.feature_dim, 512, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(),
#                 nn.Dropout2d(0.5),
#                 nn.Conv2d(512, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(256, num_classes)
#             )

#     def forward(self, x):
#         if 'vit' in self.model_name:
#             features = self.base_model(x)
#             output = self.conv_head(features)
#         else:
#             features = self.base_model.features(x) if hasattr(self.base_model, 'features') else self.base_model(x)
#             output = self.conv_head(features)
#         return output

class FaceClassifier(nn.Module):
    """Simplified face classification model with a fully connected head."""
    def __init__(self, base_model, num_classes, model_name):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        self.model_name = model_name

        # Determine the embedding dimension based on model type
        if 'efficientnet' in model_name:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
                features = base_model.features(dummy_input)
                embedding_dim = features.shape[1] * features.shape[2] * features.shape[3]
        elif 'regnet' in model_name:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
                features = base_model.features(dummy_input) if hasattr(base_model, 'features') else base_model(dummy_input)
                embedding_dim = features.shape[1] * features.shape[2] * features.shape[3]
        elif 'vit' in model_name:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
                features = base_model(dummy_input)
                embedding_dim = features.shape[-1]
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        # Define the simplified fully connected classifier head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        if 'vit' in self.model_name:
            features = self.base_model(x)
        else:
            features = self.base_model.features(x) if hasattr(self.base_model, 'features') else self.base_model(x)
        output = self.fc(features)
        return output

class FaceClassifierLightning(pl.LightningModule):
    """PyTorch Lightning module for face classification."""
    def __init__(self, base_model, num_classes, learning_rate, warmup_steps=1000, total_steps=100000, max_lr_factor=10.0, model_name='efficientnet_b0'):
        super(FaceClassifierLightning, self).__init__()
        self.model = FaceClassifier(base_model, num_classes, model_name)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = learning_rate * max_lr_factor
        self.min_lr = 1e-6
        self.model_name = model_name
        self.save_hyperparameters("num_classes", "learning_rate", "warmup_steps", "total_steps", "max_lr_factor", "model_name")

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
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_progress_bar:
            self.train_progress_bar.set_description(f"Training Epoch {trainer.current_epoch + 1}")

def main(args):
    mp.set_start_method('spawn', force=True)
    
    # Get the resolution for the selected model
    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {args.model_name} not supported. Choose from {list(MODEL_CONFIGS.keys())}")
    resolution = MODEL_CONFIGS[args.model_name]['resolution']
    
    train_cache_dir = os.path.join(args.dataset_dir, f"train_data_aligned_{args.model_name}")
    val_cache_dir = os.path.join(args.dataset_dir, f"val_data_aligned_{args.model_name}")
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
    
    # Load the appropriate model
    model_fn = MODEL_CONFIGS[args.model_name]['model_fn']
    weights = MODEL_CONFIGS[args.model_name]['weights']
    base_model = model_fn(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
    if hasattr(base_model, 'classifier'):
        base_model.classifier = nn.Identity()
    elif hasattr(base_model, 'fc'):
        base_model.fc = nn.Identity()
    elif hasattr(base_model, 'head'):
        base_model.head = nn.Identity()
    base_model.eval()
    
    model = FaceClassifierLightning(
        base_model=base_model,
        num_classes=len(train_dataset.class_to_idx),
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr_factor=args.max_lr_factor,
        model_name=args.model_name
    )
    checkpoint_callback = CustomModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename=f'face_classifier_{args.model_name}-{{epoch:02d}}-{{val_loss:.2f}}',
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
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to use for training.')

    args = parser.parse_args()
    main(args)
