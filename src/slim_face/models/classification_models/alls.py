import torch
import torch.nn as nn

class FaceClassifier(nn.Module):
    """Face classification model with a configurable head."""
    def __init__(self, base_model, num_classes, model_name):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        self.model_name = model_name
        
        # Determine the feature extraction method and output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, MODEL_CONFIGS[model_name]['resolution'], MODEL_CONFIGS[model_name]['resolution'])
            features = base_model(dummy_input)
            if len(features.shape) == 4:  # Spatial feature map (batch, channels, height, width)
                in_channels = features.shape[1]
                self.feature_type = 'spatial'
                self.feature_dim = in_channels
            elif len(features.shape) == 2:  # Flattened feature vector (batch, features)
                in_channels = features.shape[1]
                self.feature_type = 'flat'
                self.feature_dim = in_channels
            else:
                raise ValueError(f"Unexpected feature shape from base model {model_name}: {features.shape}")

        # Define the classifier head based on feature type
        if self.feature_type == 'flat' or 'vit' in model_name:
            self.conv_head = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        else:
            self.conv_head = nn.Sequential(
                nn.Conv2d(self.feature_dim, 512, kernel_size=3, padding=1),
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
        features = self.base_model(x)
        output = self.conv_head(features)
        return output