import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        # Aplana por imagen, no todo el batch junto
        preds   = preds.view(preds.size(0), -1)     # [B, H*W]
        targets = targets.view(targets.size(0), -1)  # [B, H*W]

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return (1 - dice).mean()