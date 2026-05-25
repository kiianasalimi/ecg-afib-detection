"""
preprocessing.py
----------------
Reusable ECG preprocessing functions for the AFib detection pipeline.
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=300, order=4):
    """Remove baseline wander (low-freq) and high-frequency noise."""
    nyq = fs / 2
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_rpeaks(signal, fs=300):
    """Detect R-peak positions (sample indices) using NeuroKit2."""
    signals, info = nk.ecg_process(signal, sampling_rate=fs)
    return info['ECG_R_Peaks']


def compute_rr_intervals(rpeaks, fs=300):
    """
    Convert R-peak sample indices to RR intervals in seconds.
    AFib is characterised by highly irregular RR intervals.
    """
    if len(rpeaks) < 2:
        return np.array([])
    return np.diff(rpeaks) / fs


def segment_signal(signal, fs=300, window_sec=30):
    """Split signal into fixed-length non-overlapping windows."""
    window_len = fs * window_sec
    segments = []
    for start in range(0, len(signal) - window_len, window_len):
        segments.append(signal[start:start + window_len])
    return segments


def normalize_signal(signal):
    """Zero mean, unit variance normalization per recording."""
    std = signal.std()
    if std == 0:
        return signal - signal.mean()  # flat signal edge case
    return (signal - signal.mean()) / std
