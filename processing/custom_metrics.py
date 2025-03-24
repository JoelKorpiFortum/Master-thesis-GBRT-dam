# Custom_metrics.py
import numpy as np

def willmotts_d(observed, predicted):
    numerator = np.sum((observed - predicted) ** 2)
    observed_mean = np.mean(observed)
    denominator = np.sum((np.abs(predicted - observed_mean) + np.abs(observed - observed_mean)) ** 2)
    return 1 - (numerator / denominator)

def nash_sutcliffe(observed, predicted):
    numerator = np.sum((observed - predicted) ** 2)
    observed_mean = np.mean(observed)
    denominator = np.sum((observed - observed_mean) ** 2)
    return 1 - (numerator / denominator)
