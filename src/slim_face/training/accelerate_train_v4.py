import os
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
import sys
import math
from torch.optim.lr_scheduler import LambdaLR

# Add the 'edgeface' model directory to sys.path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'edgeface')))

# Import the 'align' function from the 'face_alignment' module
from face_alignment import align

# Import the 'get_model' function from the 'backbones' module
from backbones import get_model

def preprocess_and_cache_images(input_dir, output_dir, algorithm='yolo'):
    """
    Preprocess images using YOLO-based face alignment and save aligned images to a cache directory.
    
   (No changes to this function; included for completeness)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
                    aligned_image = aligned_image.resize((224, 224), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    aligned_image = Image.open(img_path).convert('RGB')
                    aligned_image = aligned_image.resize((224, 224), Image.Resampling.LANCZOS)
                    aligned_image.save(output_img_path, quality=100)
            
            if skipped_count > 0:
                print(f"Skipped {skipped_count} images for {person} that were already processed.")

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset to load pre-aligned images.
        
        (No changes; included for completeness)
        """
        self.root_dir = root_dir
        self.transform = transform
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
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

class FaceClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        embedding = self.base_model(x)
        output = self.fc(embedding)
        return output

class FaceClassifierLightning(pl.LightningModule):
    def __init__(self, base_model, embedding_dim, num_classes, learning_rate, warmup_steps=1000, total_steps=100000):
        super(FaceClassifierLightning, self).__init__()
        self.model = FaceClassifier(base_model, embedding_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.save_hyperparameters("embedding_dim", "num_classes", "learning_rate", "warmup_steps", "total_steps")

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
        # Get current learning rate from optimizer
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f"\nEpoch {self.current_epoch + 1}: "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, "
              f"Learning rate: {current_lr:.6e}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        min_lr = 1e-6  # Minimum learning rate

        def lr_lambda(step):
            if step < self.warmup_steps:
                # During warmup, maintain the initial learning rate (5e-4)
                return 1.0
            # Cosine decay from learning_rate to min_lr
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_lr = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (self.learning_rate - min_lr) * cosine_lr
            return max(lr, min_lr) / self.learning_rate  # Normalize for LambdaLR

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

    train_cache_dir = os.path.join(args.dataset_dir, "train_data_aligned")
    val_cache_dir = os.path.join(args.dataset_dir, "val_data_aligned")

    print("Preprocessing training dataset...")
    preprocess_and_cache_images(
        input_dir=os.path.join(args.dataset_dir, "train_data"),
        output_dir=train_cache_dir,
        algorithm=args.algorithm
    )
    print("Preprocessing validation dataset...")
    preprocess_and_cache_images(
        input_dir=os.path.join(args.dataset_dir, "val_data"),
        output_dir=val_cache_dir,
        algorithm=args.algorithm
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = FaceDataset(root_dir=train_cache_dir, transform=transform)
    val_dataset = FaceDataset(root_dir=val_cache_dir, transform=transform)

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
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_steps * total_steps) if args.warmup_steps > 0 else int(0.05 * total_steps)

    model_name = os.path.basename(args.edgeface_model_path).split(".")[0]
    base_model = get_model(model_name)
    checkpoint_path = args.edgeface_model_path
    base_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    base_model.eval()

    model = FaceClassifierLightning(
        base_model=base_model,
        embedding_dim=args.embedding_dim,
        num_classes=len(train_dataset.class_to_idx),
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    checkpoint_callback = CustomModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='face_classifier-{epoch:02d}-{val_loss:.2f}',
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
    parser.add_argument('--edgeface_model_path', type=str, default='ckpts/edgeface_ckpts/edgeface_s_gamma_05.pt',
                        help='Path of the EdgeFace model.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation.')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Dimension of the embedding layer.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate for the optimizer.')
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

    args = parser.parse_args()
    main(args)
