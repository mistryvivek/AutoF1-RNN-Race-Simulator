import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedModelTransformerMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_gaussians=5, num_heads=4, num_layers=3):
        super(UnifiedModelTransformerMDN, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x).mean(dim=1)
        
        pit_output = self.fc_pit(x)
        
        pi = F.softmax(self.fc_pi(x), dim=-1)
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))
        
        return pit_output, pi, mu, sigma