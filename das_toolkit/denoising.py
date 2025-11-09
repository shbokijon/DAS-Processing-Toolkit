"""
Wavelet-based denoising methods for DAS data processing.

This module provides various wavelet denoising techniques using PyWavelets,
including soft and hard thresholding, multi-level decomposition, and
visualization tools for comparing different denoising methods.
"""

import numpy as np
import pywt
from typing import Union, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt


def estimate_noise_std(coeffs: np.ndarray, method: str = "mad") -> float:
    """
    Estimate noise standard deviation from wavelet coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Wavelet coefficients (typically finest detail coefficients)
    method : str, optional
        Method for estimation: 'mad' (Median Absolute Deviation) or 'std'
        Default is 'mad' which is more robust to outliers

    Returns
    -------
    float
        Estimated noise standard deviation

    References
    ----------
    Donoho, D. L., & Johnstone, J. M. (1994). Ideal spatial adaptation by
    wavelet shrinkage. Biometrika, 81(3), 425-455.
    """
    if method == "mad":
        # MAD estimator: sigma = MAD / 0.6745
        return np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745
    elif method == "std":
        return np.std(coeffs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mad' or 'std'")


def calculate_threshold(
    coeffs: np.ndarray,
    method: str = "universal",
    noise_std: Optional[float] = None
) -> float:
    """
    Calculate threshold value for wavelet coefficient shrinkage.

    Parameters
    ----------
    coeffs : np.ndarray
        Wavelet coefficients
    method : str, optional
        Thresholding method:
        - 'universal': Universal threshold (sigma * sqrt(2 * log(N)))
        - 'sure': Stein's Unbiased Risk Estimate
        - 'minimax': Minimax threshold
        Default is 'universal'
    noise_std : float, optional
        Noise standard deviation. If None, estimated from coefficients

    Returns
    -------
    float
        Threshold value
    """
    N = len(coeffs)

    if noise_std is None:
        noise_std = estimate_noise_std(coeffs)

    if method == "universal":
        # Universal threshold: sigma * sqrt(2 * log(N))
        return noise_std * np.sqrt(2 * np.log(N))
    elif method == "minimax":
        # Minimax threshold (simplified version)
        if N > 32:
            return noise_std * (0.3936 + 0.1829 * np.log2(N))
        else:
            return 0
    elif method == "sure":
        # SURE threshold (Stein's Unbiased Risk Estimate)
        # Sort absolute values
        abs_coeffs = np.sort(np.abs(coeffs))
        N = len(abs_coeffs)

        # Calculate risks
        risks = np.zeros(N)
        for i in range(N):
            threshold = abs_coeffs[i]
            # Count coefficients below threshold
            n_below = i + 1
            # Sum of squares above threshold
            sum_above = np.sum(abs_coeffs[i:] ** 2)
            # SURE risk
            risks[i] = (N - 2 * n_below + sum_above) / N

        # Find minimum risk
        min_idx = np.argmin(risks)
        return abs_coeffs[min_idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.

    Soft thresholding shrinks coefficients towards zero:
    - If |x| < threshold: x -> 0
    - If |x| >= threshold: x -> sign(x) * (|x| - threshold)

    Parameters
    ----------
    coeffs : np.ndarray
        Wavelet coefficients
    threshold : float
        Threshold value

    Returns
    -------
    np.ndarray
        Thresholded coefficients
    """
    return pywt.threshold(coeffs, threshold, mode='soft')


def hard_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply hard thresholding to wavelet coefficients.

    Hard thresholding keeps or removes coefficients:
    - If |x| < threshold: x -> 0
    - If |x| >= threshold: x -> x (unchanged)

    Parameters
    ----------
    coeffs : np.ndarray
        Wavelet coefficients
    threshold : float
        Threshold value

    Returns
    -------
    np.ndarray
        Thresholded coefficients
    """
    return pywt.threshold(coeffs, threshold, mode='hard')


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_method: str = 'universal',
    threshold_mode: str = 'soft',
    noise_std: Optional[float] = None,
    mode: str = 'symmetric'
) -> np.ndarray:
    """
    Denoise a 1D signal using wavelet transform with thresholding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym4', 'coif3', 'haar')
        Default is 'db4' (Daubechies 4)
    level : int, optional
        Decomposition level. If None, uses maximum useful level
    threshold_method : str, optional
        Method for calculating threshold: 'universal', 'sure', 'minimax'
        Default is 'universal'
    threshold_mode : str, optional
        Thresholding mode: 'soft' or 'hard'
        Default is 'soft'
    noise_std : float, optional
        Known noise standard deviation. If None, estimated from signal
    mode : str, optional
        Signal extension mode for wavelet transform
        Default is 'symmetric'

    Returns
    -------
    np.ndarray
        Denoised signal

    Examples
    --------
    >>> import numpy as np
    >>> # Create noisy signal
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 5 * t)
    >>> noisy_signal = signal + np.random.normal(0, 0.2, len(signal))
    >>> # Denoise
    >>> denoised = wavelet_denoise(noisy_signal, wavelet='db4', threshold_mode='soft')
    """
    # Determine decomposition level
    if level is None:
        level = pywt.dwt_max_level(len(signal), wavelet)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)

    # Estimate noise from finest detail coefficients if not provided
    if noise_std is None:
        noise_std = estimate_noise_std(coeffs[-1])

    # Apply thresholding to detail coefficients (skip approximation coefficients)
    coeffs_threshold = [coeffs[0]]  # Keep approximation coefficients

    for i in range(1, len(coeffs)):
        # Calculate threshold for this level
        threshold = calculate_threshold(
            coeffs[i],
            method=threshold_method,
            noise_std=noise_std
        )

        # Apply thresholding
        if threshold_mode == 'soft':
            coeffs_threshold.append(soft_threshold(coeffs[i], threshold))
        elif threshold_mode == 'hard':
            coeffs_threshold.append(hard_threshold(coeffs[i], threshold))
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    # Reconstruct signal
    denoised = pywt.waverec(coeffs_threshold, wavelet, mode=mode)

    # Ensure same length as input (due to numerical precision)
    return denoised[:len(signal)]


def multilevel_wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db4',
    max_level: Optional[int] = None,
    threshold_method: str = 'universal',
    threshold_mode: str = 'soft',
    mode: str = 'symmetric',
    return_all_levels: bool = False
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Perform multi-level wavelet decomposition and denoising.

    This function denoises the signal at multiple decomposition levels
    and can return either the final denoised signal or all intermediate results.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym4', 'coif3', 'haar')
    max_level : int, optional
        Maximum decomposition level. If None, uses maximum useful level
    threshold_method : str, optional
        Method for calculating threshold: 'universal', 'sure', 'minimax'
    threshold_mode : str, optional
        Thresholding mode: 'soft' or 'hard'
    mode : str, optional
        Signal extension mode for wavelet transform
    return_all_levels : bool, optional
        If True, returns dict with results for each level
        If False, returns only the final denoised signal

    Returns
    -------
    np.ndarray or dict
        If return_all_levels=False: denoised signal
        If return_all_levels=True: dict mapping level -> denoised signal

    Examples
    --------
    >>> # Get denoised signals at all levels
    >>> results = multilevel_wavelet_denoise(
    ...     noisy_signal, wavelet='db4', return_all_levels=True
    ... )
    >>> # results[1], results[2], ... contain denoised signals at each level
    """
    if max_level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)

    if return_all_levels:
        results = {}
        for level in range(1, max_level + 1):
            results[level] = wavelet_denoise(
                signal,
                wavelet=wavelet,
                level=level,
                threshold_method=threshold_method,
                threshold_mode=threshold_mode,
                mode=mode
            )
        return results
    else:
        return wavelet_denoise(
            signal,
            wavelet=wavelet,
            level=max_level,
            threshold_method=threshold_method,
            threshold_mode=threshold_mode,
            mode=mode
        )


def compare_denoising_methods(
    signal: np.ndarray,
    wavelets: Optional[List[str]] = None,
    threshold_modes: Optional[List[str]] = None,
    threshold_methods: Optional[List[str]] = None,
    level: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compare different denoising methods on the same signal.

    Parameters
    ----------
    signal : np.ndarray
        Input noisy signal
    wavelets : list of str, optional
        List of wavelet types to compare.
        Default is ['db4', 'sym4', 'coif3', 'haar']
    threshold_modes : list of str, optional
        List of threshold modes to compare.
        Default is ['soft', 'hard']
    threshold_methods : list of str, optional
        List of threshold methods to compare.
        Default is ['universal']
    level : int, optional
        Decomposition level. If None, uses maximum useful level

    Returns
    -------
    dict
        Dictionary mapping method name to denoised signal
        Keys are formatted as: 'wavelet_mode_method'

    Examples
    --------
    >>> results = compare_denoising_methods(
    ...     noisy_signal,
    ...     wavelets=['db4', 'haar'],
    ...     threshold_modes=['soft', 'hard']
    ... )
    >>> # results contains: 'db4_soft_universal', 'db4_hard_universal', etc.
    """
    if wavelets is None:
        wavelets = ['db4', 'sym4', 'coif3', 'haar']
    if threshold_modes is None:
        threshold_modes = ['soft', 'hard']
    if threshold_methods is None:
        threshold_methods = ['universal']

    results = {}

    for wavelet in wavelets:
        for mode in threshold_modes:
            for method in threshold_methods:
                key = f"{wavelet}_{mode}_{method}"
                results[key] = wavelet_denoise(
                    signal,
                    wavelet=wavelet,
                    level=level,
                    threshold_method=method,
                    threshold_mode=mode
                )

    return results


def visualize_denoising_comparison(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised_signals: Dict[str, np.ndarray],
    title: str = "Wavelet Denoising Comparison",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize and compare different denoising results.

    Creates a comprehensive visualization showing:
    - Original and noisy signals
    - Multiple denoised signals
    - Error plots
    - SNR/RMSE metrics

    Parameters
    ----------
    original : np.ndarray
        Original clean signal (for reference)
    noisy : np.ndarray
        Noisy signal
    denoised_signals : dict
        Dictionary mapping method name to denoised signal
    title : str, optional
        Figure title
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    matplotlib.figure.Figure
        The created figure

    Examples
    --------
    >>> results = compare_denoising_methods(noisy_signal, wavelets=['db4', 'haar'])
    >>> fig = visualize_denoising_comparison(
    ...     clean_signal, noisy_signal, results
    ... )
    >>> plt.show()
    """
    n_methods = len(denoised_signals)
    n_rows = n_methods + 2  # Original/noisy + denoised methods + metrics

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Time axis
    t = np.arange(len(original))

    # Calculate metrics for noisy signal
    noisy_snr = calculate_snr(original, noisy)
    noisy_rmse = calculate_rmse(original, noisy)

    # Plot 1: Original and noisy signals
    axes[0, 0].plot(t, original, 'b-', label='Original', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(t, noisy, 'r-', label='Noisy', alpha=0.5, linewidth=0.8)
    axes[0, 0].set_title(f'Original vs Noisy (SNR: {noisy_snr:.2f} dB, RMSE: {noisy_rmse:.4f})')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Noise
    noise = noisy - original
    axes[0, 1].plot(t, noise, 'k-', alpha=0.5, linewidth=0.8)
    axes[0, 1].set_title(f'Noise (std: {np.std(noise):.4f})')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot denoised signals
    metrics = []
    for idx, (method_name, denoised) in enumerate(denoised_signals.items(), start=1):
        # Calculate metrics
        snr = calculate_snr(original, denoised)
        rmse = calculate_rmse(original, denoised)
        metrics.append((method_name, snr, rmse))

        # Left: Signal comparison
        axes[idx, 0].plot(t, original, 'b-', label='Original', alpha=0.5, linewidth=1)
        axes[idx, 0].plot(t, denoised, 'g-', label='Denoised', alpha=0.8, linewidth=1.2)
        axes[idx, 0].set_title(f'{method_name} (SNR: {snr:.2f} dB, RMSE: {rmse:.4f})')
        axes[idx, 0].set_xlabel('Sample')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Right: Error
        error = denoised - original
        axes[idx, 1].plot(t, error, 'r-', alpha=0.6, linewidth=0.8)
        axes[idx, 1].set_title(f'{method_name} Error (std: {np.std(error):.4f})')
        axes[idx, 1].set_xlabel('Sample')
        axes[idx, 1].set_ylabel('Error')
        axes[idx, 1].grid(True, alpha=0.3)

    # Summary metrics plot
    if n_methods > 0:
        methods = [m[0] for m in metrics]
        snrs = [m[1] for m in metrics]
        rmses = [m[2] for m in metrics]

        # SNR comparison
        axes[-1, 0].barh(methods, snrs, color='steelblue', alpha=0.7)
        axes[-1, 0].axvline(noisy_snr, color='r', linestyle='--',
                           label=f'Noisy: {noisy_snr:.2f} dB')
        axes[-1, 0].set_xlabel('SNR (dB)')
        axes[-1, 0].set_title('Signal-to-Noise Ratio Comparison')
        axes[-1, 0].legend()
        axes[-1, 0].grid(True, alpha=0.3, axis='x')

        # RMSE comparison
        axes[-1, 1].barh(methods, rmses, color='coral', alpha=0.7)
        axes[-1, 1].axvline(noisy_rmse, color='r', linestyle='--',
                           label=f'Noisy: {noisy_rmse:.4f}')
        axes[-1, 1].set_xlabel('RMSE')
        axes[-1, 1].set_title('Root Mean Square Error Comparison')
        axes[-1, 1].legend()
        axes[-1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def calculate_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in decibels.

    Parameters
    ----------
    clean_signal : np.ndarray
        Original clean signal
    noisy_signal : np.ndarray
        Noisy or denoised signal

    Returns
    -------
    float
        SNR in decibels
    """
    noise = noisy_signal - clean_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_rmse(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Parameters
    ----------
    clean_signal : np.ndarray
        Original clean signal
    noisy_signal : np.ndarray
        Noisy or denoised signal

    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(np.mean((clean_signal - noisy_signal) ** 2))


def denoise_2d_signal(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_method: str = 'universal',
    threshold_mode: str = 'soft',
    axis: int = -1,
    mode: str = 'symmetric'
) -> np.ndarray:
    """
    Denoise a 2D signal (e.g., DAS data matrix) along a specified axis.

    This is useful for DAS data where you have time series for multiple channels.
    You can denoise along the time axis (axis=1) or channel axis (axis=0).

    Parameters
    ----------
    signal : np.ndarray
        Input 2D signal (e.g., channels × time samples)
    wavelet : str, optional
        Wavelet type
    level : int, optional
        Decomposition level
    threshold_method : str, optional
        Threshold calculation method
    threshold_mode : str, optional
        'soft' or 'hard' thresholding
    axis : int, optional
        Axis along which to denoise (default is -1, last axis)
    mode : str, optional
        Signal extension mode

    Returns
    -------
    np.ndarray
        Denoised 2D signal

    Examples
    --------
    >>> # DAS data: 100 channels × 10000 time samples
    >>> das_data = np.random.randn(100, 10000)
    >>> # Denoise along time axis
    >>> denoised = denoise_2d_signal(das_data, axis=1)
    """
    # Apply 1D denoising along specified axis
    denoised = np.apply_along_axis(
        lambda x: wavelet_denoise(
            x,
            wavelet=wavelet,
            level=level,
            threshold_method=threshold_method,
            threshold_mode=threshold_mode,
            mode=mode
        ),
        axis=axis,
        arr=signal
    )

    return denoised
