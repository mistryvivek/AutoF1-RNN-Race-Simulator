import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(UnifiedModelRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
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
        _, (h_n, _) = self.rnn(x)
        h_n = h_n.squeeze(0)
        
        pit_output = self.fc_pit(h_n)
        time_output = self.fc_time(h_n)
        
        return pit_output, time_output