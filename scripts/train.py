import torch
import torch.nn as nn
import torchvision.transforms as transforms
from edgeface.face_alignment import align
from edgeface.backbones import get_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import warnings

# Custom Dataset class for loading face images
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Suppress FutureWarning from numpy.linalg.lstsq during dataset initialization
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

        # Suppress FutureWarning from numpy.linalg.lstsq during image loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*rcond.*")
            try:
                # Load image and resize immediately to ensure consistent size
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

                # Attempt face alignment
                aligned = align.get_aligned_face(img_path)[1]
                # aligned = aligned[1]

                if aligned is None:
                    print(f"Face detection failed for {img_path}, using resized original image")
                    aligned = image
                else:
                    # Ensure aligned image is 224x224
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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({'loss': loss.item()})

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix({'loss': loss.item()})

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
batch_size = 64
embedding_dim = 512
num_epochs = 100
learning_rate = 1e-3

# Load base model
model_name = "edgeface_s_gamma_05"
base_model = get_model(model_name)
checkpoint_path = f'./edgeface/checkpoints/{model_name}.pt'
base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
base_model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load datasets
main_dataset_dir = "./dataset/processed_ds"

train_dataset = FaceDataset(root_dir=os.path.join(main_dataset_dir, "train_data"),
                           transform=transform)
val_dataset = FaceDataset(root_dir=os.path.join(main_dataset_dir, "val_data"),
                         transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize classifier
num_classes = len(train_dataset.class_to_idx)
model = FaceClassifier(base_model, embedding_dim, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

# Train the model using the function
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

