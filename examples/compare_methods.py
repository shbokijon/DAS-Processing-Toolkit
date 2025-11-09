"""
Compare Different Denoising Methods

This example demonstrates how to compare different wavelet denoising
methods and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from das_toolkit import (
    compare_denoising_methods,
    visualize_denoising_comparison,
    wavelet_denoise,
)

# Set random seed for reproducibility
np.random.seed(42)

# Create a clean signal
print("Generating test signal...")
t = np.linspace(0, 2, 2000)
clean_signal = (
    np.sin(2 * np.pi * 3 * t) +
    0.7 * np.sin(2 * np.pi * 7 * t) +
    0.4 * np.sin(2 * np.pi * 15 * t)
)

# Add noise
noise_level = 0.4
noisy_signal = clean_signal + np.random.normal(0, noise_level, len(clean_signal))

print(f"Added Gaussian noise with std={noise_level}")

# Compare different wavelet families
print("\nComparing different wavelet families...")
wavelets = ['db4', 'sym4', 'coif3', 'haar']
results_wavelets = compare_denoising_methods(
    noisy_signal,
    wavelets=wavelets,
    threshold_modes=['soft'],
    threshold_methods=['universal']
)

# Compare soft vs hard thresholding
print("Comparing soft vs hard thresholding...")
results_thresholding = compare_denoising_methods(
    noisy_signal,
    wavelets=['db4'],
    threshold_modes=['soft', 'hard'],
    threshold_methods=['universal']
)

# Compare threshold methods
print("Comparing threshold methods...")
results_threshold_methods = compare_denoising_methods(
    noisy_signal,
    wavelets=['db4'],
    threshold_modes=['soft'],
    threshold_methods=['universal', 'sure', 'minimax']
)

# Visualize wavelet comparison
print("\nGenerating comparison visualizations...")
fig1 = visualize_denoising_comparison(
    clean_signal,
    noisy_signal,
    results_wavelets,
    title="Wavelet Family Comparison",
    figsize=(15, 10),
    save_path='examples/wavelet_family_comparison.png'
)

# Visualize thresholding mode comparison
fig2 = visualize_denoising_comparison(
    clean_signal,
    noisy_signal,
    results_thresholding,
    title="Soft vs Hard Thresholding Comparison",
    figsize=(15, 8),
    save_path='examples/thresholding_mode_comparison.png'
)

# Visualize threshold method comparison
fig3 = visualize_denoising_comparison(
    clean_signal,
    noisy_signal,
    results_threshold_methods,
    title="Threshold Method Comparison",
    figsize=(15, 10),
    save_path='examples/threshold_method_comparison.png'
)

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("="*60)
print("Saved visualizations:")
print("  1. examples/wavelet_family_comparison.png")
print("  2. examples/thresholding_mode_comparison.png")
print("  3. examples/threshold_method_comparison.png")
print("="*60)

plt.show()
