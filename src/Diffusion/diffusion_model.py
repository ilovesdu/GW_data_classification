import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + skip)

class DiffusionModel(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=64):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(input_channels, hidden_channels),
            nn.MaxPool1d(2),
            ResidualBlock(hidden_channels, hidden_channels * 2)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_channels * 2, hidden_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        return self.decoder(self.encoder(x))

if __name__ == "__main__":
    model = DiffusionModel()
    x = torch.randn(8, 2, 8192)  # batch_size=8
    t = torch.tensor([10] * 8)  # 时间步
    print(model(x, t).shape)  # (8, 2, 8192)
