import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.2, pos_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        # Calculate BCE Loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate probability of the positive class
        p_t = torch.exp(-bce_loss)  # p_t = exp(-loss)

        # Apply focal loss factor
        loss = self.alpha * (1 - p_t)**self.gamma * bce_loss
        return loss.mean()  # We take the mean loss over all batches
