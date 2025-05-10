import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.network = self.make_layers([
            ('conv', 32),
            ('conv', 64),
            ('max_pool', None),
            ('batch_norm', None),
            ('conv', 128),
            ('conv', 128),
            ('max_pool', None),
            ('batch_norm', None),
            ('conv', 256),
            ('conv', 256),
            ('max_pool', None),
            ('batch_norm', None),
            ('flatten', 4096),
            ('linear', 1024),
            ('linear', 512),
        ], z_dim)

    def make_layers(self, structure, z_dim):
        layers = []
        in_channels = 3
        for layer, param in structure:
            if layer == 'max_pool':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer == 'conv':
                layers += [
                           nn.Conv2d(in_channels, param, kernel_size=3, stride=1, padding=1),
                           nn.ReLU()
                        ]
                in_channels = param
            elif layer == 'batch_norm':
                layers += [nn.BatchNorm2d(in_channels)]
            elif layer == 'flatten':
                layers += [nn.Flatten()]
                in_channels = param
            else:
                layers += [
                    nn.Linear(in_channels, param),
                    nn.ReLU()
                ]
                in_channels = param
        layers += [nn.Linear(in_channels, z_dim)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
  
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)