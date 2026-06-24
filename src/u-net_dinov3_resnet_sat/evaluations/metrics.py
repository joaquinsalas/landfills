# evaluations/metrics.py
# ─────────────────────────────────────────────────────────────────────
# Funciones de metricas de segmentacion:
# IoU, Dice, Precision, Recall, F1, BCE Loss.
# ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def _binarize(pred, thr=0.5):
    """Convierte prediccion continua a mascara binaria con umbral."""
    return (pred > thr).float()


def iou(pred, target, thr=0.5):
    """
    Intersection over Union.
    Mide superposicion entre prediccion y mascara real.
    Rango: 0 (nada) a 1 (perfecto).
    """
    p     = _binarize(pred, thr)
    inter = (p * target).sum()
    union = (p + target).clamp(0, 1).sum()
    return (inter / (union + 1e-8)).item()


def dice(pred, target, thr=0.5):
    """
    Dice Score (F1 espacial).
    Mas sensible que IoU en clases desbalanceadas.
    Rango: 0 a 1.
    """
    p     = _binarize(pred, thr)
    inter = (p * target).sum()
    return (2 * inter / (p.sum() + target.sum() + 1e-8)).item()


def precision(pred, target, thr=0.5):
    """
    Precision: de lo predicho como relleno, cuanto realmente lo era.
    Alta precision = pocas falsas alarmas.
    """
    p  = _binarize(pred, thr)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    return (tp / (tp + fp + 1e-8)).item()


def recall(pred, target, thr=0.5):
    """
    Recall: de todo el relleno real, cuanto detecto el modelo.
    Alto recall = pocos rellenos sin detectar.
    Critico para aplicaciones ambientales.
    """
    p  = _binarize(pred, thr)
    tp = (p * target).sum()
    fn = ((1 - p) * target).sum()
    return (tp / (tp + fn + 1e-8)).item()


def f1_score(prec, rec):
    """
    F1 Score: balance armonico entre precision y recall.
    Metrica mas equilibrada para evaluacion general.
    """
    return 2 * (prec * rec) / (prec + rec + 1e-8)


def bce_loss(pred, target):
    """BCE Loss pixel a pixel entre prediccion y mascara real."""
    return nn.BCELoss()(pred, target).item()


def calcular_todas(pred, target, thr=0.5):
    """
    Calcula todas las metricas de una vez.
    Retorna diccionario con los valores.
    """
    pr = precision(pred, target, thr)
    rc = recall(pred, target, thr)
    return {
        'iou':       iou(pred, target, thr),
        'dice':      dice(pred, target, thr),
        'precision': pr,
        'recall':    rc,
        'f1':        f1_score(pr, rc),
        'loss':      bce_loss(pred, target),
    }
