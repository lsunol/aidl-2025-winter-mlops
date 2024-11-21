import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, hidden_size=128, dropout=0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque de capas lineales
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_size),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 15) # 15 -> output classes in the dataset
        )

    def forward(self, x):
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
