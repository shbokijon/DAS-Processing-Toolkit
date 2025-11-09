"""
Basic Wavelet Denoising Example

This example demonstrates basic wavelet denoising on a synthetic signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from das_toolkit import wavelet_denoise, calculate_snr, calculate_rmse

# Set random seed for reproducibility
np.random.seed(42)

# Create a clean signal (combination of sine waves)
print("Generating test signal...")
t = np.linspace(0, 1, 1000)
clean_signal = (
    np.sin(2 * np.pi * 5 * t) +
    0.5 * np.sin(2 * np.pi * 10 * t) +
    0.3 * np.sin(2 * np.pi * 20 * t)
)

# Add Gaussian noise
noise_level = 0.5
noisy_signal = clean_signal + np.random.normal(0, noise_level, len(clean_signal))

# Perform wavelet denoising
print("Applying wavelet denoising...")
denoised_signal = wavelet_denoise(
    noisy_signal,
    wavelet='db4',           # Daubechies 4 wavelet
    threshold_mode='soft',    # Soft thresholding
    threshold_method='universal'
)

# Calculate metrics
snr_noisy = calculate_snr(clean_signal, noisy_signal)
snr_denoised = calculate_snr(clean_signal, denoised_signal)
rmse_noisy = calculate_rmse(clean_signal, noisy_signal)
rmse_denoised = calculate_rmse(clean_signal, denoised_signal)

# Print results
print("\n" + "="*50)
print("DENOISING RESULTS")
print("="*50)
print(f"Input noise level: {noise_level:.4f}")
print(f"\nNoisy signal:")
print(f"  SNR:  {snr_noisy:.2f} dB")
print(f"  RMSE: {rmse_noisy:.4f}")
print(f"\nDenoised signal:")
print(f"  SNR:  {snr_denoised:.2f} dB")
print(f"  RMSE: {rmse_denoised:.4f}")
print(f"\nImprovement:")
print(f"  SNR:  +{snr_denoised - snr_noisy:.2f} dB")
print(f"  RMSE: -{rmse_noisy - rmse_denoised:.4f}")
print("="*50)

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot 1: Clean signal
axes[0].plot(t, clean_signal, 'b-', linewidth=1.5)
axes[0].set_title('Original Clean Signal')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Plot 2: Noisy signal
axes[1].plot(t, noisy_signal, 'r-', alpha=0.6, linewidth=0.8)
axes[1].plot(t, clean_signal, 'b-', alpha=0.3, linewidth=1.5, label='Clean')
axes[1].set_title(f'Noisy Signal (SNR: {snr_noisy:.2f} dB)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Denoised signal
axes[2].plot(t, denoised_signal, 'g-', linewidth=1.5, label='Denoised')
axes[2].plot(t, clean_signal, 'b--', alpha=0.5, linewidth=1, label='Clean')
axes[2].set_title(f'Denoised Signal (SNR: {snr_denoised:.2f} dB)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/basic_denoising_result.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: examples/basic_denoising_result.png")
plt.show()
