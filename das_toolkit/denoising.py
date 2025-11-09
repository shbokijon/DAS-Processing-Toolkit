"""
Denoising algorithms for DAS data processing.
"""

import numpy as np
import pywt
from scipy import signal


def wavelet_denoise(data, wavelet='db4', level=None, threshold_mode='soft'):
    """
    Denoise data using wavelet thresholding.

    Parameters
    ----------
    data : ndarray
        Input signal data (1D or 2D)
    wavelet : str, optional
        Wavelet type (default: 'db4')
    level : int, optional
        Decomposition level (default: None, auto-computed)
    threshold_mode : str, optional
        Thresholding mode: 'soft' or 'hard' (default: 'soft')

    Returns
    -------
    ndarray
        Denoised signal

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if data.size == 0:
        raise ValueError("Input data cannot be empty")

    if threshold_mode not in ['soft', 'hard']:
        raise ValueError("threshold_mode must be 'soft' or 'hard'")

    # Handle 1D data
    if data.ndim == 1:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        coeffs_thresh = [coeffs[0]]
        for coeff in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(coeff, threshold, mode=threshold_mode))

        return pywt.waverec(coeffs_thresh, wavelet)

    # Handle 2D data
    elif data.ndim == 2:
        coeffs = pywt.wavedec2(data, wavelet, level=level)

        # Calculate threshold from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(data.size))

        # Threshold all detail coefficients
        coeffs_thresh = [coeffs[0]]
        for detail_coeffs in coeffs[1:]:
            coeffs_thresh.append(
                tuple(pywt.threshold(c, threshold, mode=threshold_mode) for c in detail_coeffs)
            )

        return pywt.waverec2(coeffs_thresh, wavelet)

    else:
        raise ValueError("Data must be 1D or 2D array")


def fk_filter(data, dx, dt, vmin, vmax):
    """
    Apply frequency-wavenumber (FK) filtering to remove noise.

    Parameters
    ----------
    data : ndarray
        2D input data (time x space)
    dx : float
        Spatial sampling interval
    dt : float
        Temporal sampling interval
    vmin : float
        Minimum velocity to keep
    vmax : float
        Maximum velocity to keep

    Returns
    -------
    ndarray
        Filtered data

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    if dx <= 0 or dt <= 0:
        raise ValueError("Sampling intervals must be positive")
    if vmin <= 0 or vmax <= 0:
        raise ValueError("Velocities must be positive")
    if vmin >= vmax:
        raise ValueError("vmin must be less than vmax")

    # Perform 2D FFT
    fk_spectrum = np.fft.fft2(data)
    fk_spectrum_shifted = np.fft.fftshift(fk_spectrum)

    # Create frequency and wavenumber axes
    nt, nx = data.shape
    freq = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    knum = np.fft.fftshift(np.fft.fftfreq(nx, dx))

    # Create 2D meshgrid
    K, F = np.meshgrid(knum, freq)

    # Create FK filter mask
    mask = np.zeros_like(fk_spectrum_shifted)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity = np.abs(F / (K + 1e-10))

    # Keep velocities within the specified range
    mask[(velocity >= vmin) & (velocity <= vmax)] = 1

    # Apply filter and inverse FFT
    fk_filtered = fk_spectrum_shifted * mask
    fk_filtered_unshifted = np.fft.ifftshift(fk_filtered)
    filtered_data = np.real(np.fft.ifft2(fk_filtered_unshifted))

    return filtered_data


def svd_denoise(data, n_components=None, threshold=0.1):
    """
    Denoise 2D data using Singular Value Decomposition (SVD).

    Parameters
    ----------
    data : ndarray
        2D input data
    n_components : int, optional
        Number of components to keep (default: None, auto-select)
    threshold : float, optional
        Threshold for auto-selecting components (default: 0.1)

    Returns
    -------
    ndarray
        Denoised data

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    if data.size == 0:
        raise ValueError("Input data cannot be empty")
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Perform SVD
    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    # Auto-select number of components if not specified
    if n_components is None:
        # Keep components above threshold of max singular value
        n_components = np.sum(s > threshold * s[0])
        n_components = max(1, n_components)  # Keep at least one component

    if n_components < 1 or n_components > min(data.shape):
        raise ValueError(f"n_components must be between 1 and {min(data.shape)}")

    # Reconstruct with limited components
    s_truncated = s.copy()
    s_truncated[n_components:] = 0

    denoised = U @ np.diag(s_truncated) @ Vt

    return denoised


def median_denoise(data, kernel_size=(3, 3)):
    """
    Denoise 2D data using median filtering.

    Parameters
    ----------
    data : ndarray
        2D input data
    kernel_size : tuple, optional
        Size of the median filter kernel (default: (3, 3))

    Returns
    -------
    ndarray
        Denoised data

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    if data.size == 0:
        raise ValueError("Input data cannot be empty")

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise ValueError("kernel_size must be a tuple of length 2")

    if kernel_size[0] < 1 or kernel_size[1] < 1:
        raise ValueError("Kernel dimensions must be at least 1")

    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    return signal.medfilt2d(data, kernel_size=kernel_size)
