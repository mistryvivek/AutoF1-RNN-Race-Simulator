import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedModelCNNRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(UnifiedModelCNNRNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.rnn = nn.LSTM(128, hidden_dim, batch_first=True)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.permute(0, 2, 1)
        
        _, (h_n, _) = self.rnn(x)
        h_n = h_n.squeeze(0)
        
        pit_output = self.fc_pit(h_n)
        time_output = self.fc_time(h_n)
        
        return pit_output, time_output