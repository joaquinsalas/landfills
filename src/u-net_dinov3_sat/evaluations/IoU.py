import torch

def calculate_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold
    intersection = (preds & masks).float().sum((1, 2, 3))
    union = (preds | masks).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
