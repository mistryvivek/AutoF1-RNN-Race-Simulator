import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryPolyLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        """
        PolyLoss for Binary Classification
        Args:
            epsilon (float): Weight for the polynomial term. Default is 1.0.
        """
        super(BinaryPolyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Computes Binary PolyLoss.
        Args:
            logits: Predicted logits from the model (before sigmoid).
            targets: Ground truth labels (binary values: 0 or 1).
        Returns:
            loss: PolyLoss value.
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # Standard BCE Loss
        poly_term = (1 - torch.abs(targets - probs)) ** 2  # Polynomial adjustment
        loss = ce_loss + self.epsilon * poly_term  # Combine loss terms
        return loss.mean()
