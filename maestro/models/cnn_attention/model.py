from torch import nn
import torch.nn.functional as F
import torch


# To call this class, use the following code:
"""
model = StockEmbedding(output_size=64)
x = torch.randn(32, 3, 120) # 64 samples of 10-dimensional time series with length 100
#x = x.unsqueeze(1)

print(x.shape)
x = model(x)
print(x.shape)
"""
class StockEmbedding(nn.Module):
    def __init__(self, output_size, ):
        input_size = 3
        input_sequence_length = 120
        super(StockEmbedding, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=(1,1)) # batch_size*32*3*120
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1,1))
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1,1))
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1,1))
        self.fc1 = nn.Linear(128*input_size*input_sequence_length, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # Add a channel dimension for CNN
        x = x.unsqueeze(1)
        
        # First CNN layer
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Second CNN layer
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        # Third CNN layer
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x