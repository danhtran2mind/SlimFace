import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edgeface.face_alignment import align
from edgeface.backbones import get_model

# Custom Dataset class for loading face images
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                aligned = align.get_aligned_face(img_path)[1]

                if aligned is None:
                    print(f"Face detection failed for {img_path}, using resized original image")
                    aligned = image
                else:
                    aligned = aligned.resize((224, 224), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                aligned = Image.new('RGB', (224, 224))

        if self.transform:
            aligned = self.transform(aligned)

        return aligned, label

# Define the classification model
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

# PyTorch Lightning module for training
class FaceClassifierLightning(pl.LightningModule):
    def __init__(self, base_model, embedding_dim, num_classes, learning_rate):
        super(FaceClassifierLightning, self).__init__()
        self.model = FaceClassifier(base_model, embedding_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        return optimizer

def main(args):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load datasets
    train_dataset = FaceDataset(root_dir=os.path.join(args.dataset_dir, "train_data"), transform=transform)
    val_dataset = FaceDataset(root_dir=os.path.join(args.dataset_dir, "val_data"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)

    # Load base model
    base_model = get_model(args.model_name)
    checkpoint_path = f'./edgeface/checkpoints/{args.model_name}.pt'
    base_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    base_model.eval()

    # Initialize Lightning module
    model = FaceClassifierLightning(
        base_model=base_model,
        embedding_dim=args.embedding_dim,
        num_classes=len(train_dataset.class_to_idx),
        learning_rate=args.learning_rate
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='face_classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    progress_bar = TQDMProgressBar()

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a face classification model with PyTorch Lightning.')
    parser.add_argument('--dataset_dir', type=str, default='./data/processed_ds',
                        help='Path to the dataset directory.')
    parser.add_argument('--model_name', type=str, default='edgeface_s_gamma_05',
                        help='Name of the base EdgeFace model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation.')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Dimension of the embedding layer.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['cpu', 'gpu', 'tpu', 'auto'],
                        help='Accelerator type for training.')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use (e.g., number of GPUs).')

    args = parser.parse_args()
    main(args)
