import torch
import torch.nn as nn
import torch.nn.functional as F
#(8192,2)input
#0/1 output
class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super(CNNLSTM, self).__init__()
        
        # Path 1: Full signal
        self.cnn1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        
        # Path 2: Masked signal
        self.cnn2 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid()
            # 移除了 Sigmoid 层
        )
    
    def forward(self, x):
        # Path 1
        x1 = self.cnn1(x.permute(0, 2, 1))  # Convert to (batch, channel, time)
        x1, _ = self.lstm(x1.permute(0, 2, 1))  # Convert back to (batch, time, channel)
        x1 = x1[:, -1, :]  # Take the last time step
        
        # Path 2
        x2 = x[:, 3072:5120, :]  # Mask to focus on the central region
        x2 = self.cnn2(x2.permute(0, 2, 1))
        x2 = torch.mean(x2, dim=2)  # Global average pooling
        
        # Combine paths
        combined = torch.cat((x1, x2), dim=1)
        output = self.classifier(combined)
        return output  # 输出未经过 Sigmoid
