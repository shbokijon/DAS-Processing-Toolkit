"""
Signal filtering functions for DAS data processing.
"""

import numpy as np
from scipy import signal


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters
    ----------
    data : ndarray
        Input signal data
    lowcut : float
        Low frequency cutoff (Hz)
    highcut : float
        High frequency cutoff (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int, optional
        Filter order (default: 5)

    Returns
    -------
    ndarray
        Filtered signal

    Raises
    ------
    ValueError
        If frequency parameters are invalid
    """
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive")
    if lowcut >= highcut:
        raise ValueError("Low cutoff must be less than high cutoff")
    if highcut >= fs / 2:
        raise ValueError("High cutoff must be less than Nyquist frequency")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth lowpass filter to the data.

    Parameters
    ----------
    data : ndarray
        Input signal data
    cutoff : float
        Cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int, optional
        Filter order (default: 5)

    Returns
    -------
    ndarray
        Filtered signal

    Raises
    ------
    ValueError
        If frequency parameters are invalid
    """
    if cutoff <= 0:
        raise ValueError("Cutoff frequency must be positive")
    if cutoff >= fs / 2:
        raise ValueError("Cutoff must be less than Nyquist frequency")

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def highpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth highpass filter to the data.

    Parameters
    ----------
    data : ndarray
        Input signal data
    cutoff : float
        Cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int, optional
        Filter order (default: 5)

    Returns
    -------
    ndarray
        Filtered signal

    Raises
    ------
    ValueError
        If frequency parameters are invalid
    """
    if cutoff <= 0:
        raise ValueError("Cutoff frequency must be positive")
    if cutoff >= fs / 2:
        raise ValueError("Cutoff must be less than Nyquist frequency")

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(order, normal_cutoff, btype='high')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def median_filter(data, kernel_size=5):
    """
    Apply a median filter to the data.

    Parameters
    ----------
    data : ndarray
        Input signal data
    kernel_size : int, optional
        Size of the median filter kernel (default: 5)

    Returns
    -------
    ndarray
        Filtered signal

    Raises
    ------
    ValueError
        If kernel_size is invalid
    """
    if kernel_size < 1:
        raise ValueError("Kernel size must be at least 1")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    return signal.medfilt(data, kernel_size=kernel_size)


def moving_average_filter(data, window_size=5):
    """
    Apply a moving average filter to the data.

    Parameters
    ----------
    data : ndarray
        Input signal data
    window_size : int, optional
        Size of the moving average window (default: 5)

    Returns
    -------
    ndarray
        Filtered signal

    Raises
    ------
    ValueError
        If window_size is invalid
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    if window_size > len(data):
        raise ValueError("Window size cannot exceed data length")

    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')
