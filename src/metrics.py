import torch


def segmentation_metrics(logits, targets, thresh=0.5):
    pred = (torch.sigmoid(logits) > thresh).float()
    targets = (targets > 0.5).float()
    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
