# Custom_metrics.py
import numpy as np
import scipy.stats as scs

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

def kling_gupta(observed, predicted):
    r = scs.pearsonr(observed, predicted)[0]  # correlation
    alpha = (np.mean(predicted)/np.mean(observed))  # bias ratio
    beta = (np.var(predicted)/np.var(observed))  # variability ratio
    return 1 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)