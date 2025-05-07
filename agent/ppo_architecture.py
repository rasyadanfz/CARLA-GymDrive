import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torchvision.models as models

# Define the continuous action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

class CustomExtractor_PPO_End2end(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dim = 512
        self.lidar_dim = 256
        self.others_dim = 64
        features_dim = self.image_dim + self.lidar_dim + self.others_dim

        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Custom CNN for processing the RGB data
        self.rgb_network = models.resnet18(pretrained=True)
        self.rgb_network.fc = nn.Identity()

        # Custom CNN for processing the LiDAR data
        self.lidar_network = nn.Sequential(
            nn.Linear(3 * 1000, 256),
            nn.ReLU(),
        )

        # Define the neural network architecture for processing the rest of the input
        self.others_network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, lidar_input, rest_input = self.process_observations(observations)
        lidar_input = lidar_input.view(lidar_input.size(0), -1)  # Flatten the LiDAR data
                
        image_features = self.rgb_network(rgb_input)
        image_features = torch.flatten(image_features, 1)

        lidar_output = self.lidar_network(lidar_input)
        rest_output = self.others_network(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((image_features, lidar_output, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        lidar_data = observations['lidar_data']
        # Normalize each coordinate axis (X, Y, Z) individually
        lidar_data = (lidar_data - lidar_data.mean(dim=1, keepdim=True)) / (lidar_data.std(dim=1, keepdim=True) + 1e-6)

        lidar_data = lidar_data.to(self.device)
        others_data = observations['others'].float()

        return (rgb_data.float().to(self.device), lidar_data.to(self.device), others_data.to(self.device))


class CustomExtractor_PPO_Modular(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Compute the combined feature dimension
        image_dim = 1280  # Dimensionality of the EfficientNet features
        rest_dim = 256   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Load EfficientNet model from NVIDIA Torch Hub
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.efficientnet.eval()
        # Freeze the EfficientNet parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False  

        # Add adaptive average pooling to reduce the spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Define the neural network architecture for processing the rest of the input
        # 3 -> 256
        self.rest_model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((rgb_input, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        rgb_data = self.efficientnet(rgb_data)
        rgb_data = self.global_avg_pool(rgb_data)
        rgb_data = torch.flatten(rgb_data, 1).squeeze(0)

        return (rgb_data, observations['rest'])