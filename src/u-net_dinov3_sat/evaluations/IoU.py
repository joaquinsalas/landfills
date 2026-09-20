import torch

def calculate_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold
    intersection = (preds & masks).float().sum((1, 2, 3))
    union = (preds | masks).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def calculate_metrics(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold

    tp = (preds & masks).float().sum((1, 2, 3))
    fp = (preds & ~masks).float().sum((1, 2, 3))
    fn = (~preds & masks).float().sum((1, 2, 3))

    intersection = tp
    union = tp + fp + fn

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
    }
