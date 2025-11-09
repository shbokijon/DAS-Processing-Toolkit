"""
DAS Data Denoising Example

This example demonstrates wavelet denoising on 2D DAS (Distributed Acoustic
Sensing) data, where we have multiple channels recording over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from das_toolkit.denoising import denoise_2d_signal, calculate_snr, calculate_rmse

# Set random seed for reproducibility
np.random.seed(42)

# Simulate DAS data parameters
n_channels = 50        # Number of spatial channels
n_samples = 2000       # Number of time samples
noise_level = 0.3      # Noise standard deviation

print("Generating synthetic DAS data...")
print(f"  Channels: {n_channels}")
print(f"  Time samples: {n_samples}")
print(f"  Noise level: {noise_level}")

# Create synthetic DAS data (channels × time)
# Simulate a wave propagating across channels
t = np.linspace(0, 2, n_samples)
clean_das_data = np.zeros((n_channels, n_samples))

# Create a propagating wave with some spatial variation
for i in range(n_channels):
    # Each channel has slightly different frequency content
    channel_delay = i * 0.02  # Propagation delay
    clean_das_data[i, :] = (
        np.sin(2 * np.pi * 5 * (t - channel_delay)) +
        0.5 * np.sin(2 * np.pi * 10 * (t - channel_delay)) +
        0.3 * np.cos(2 * np.pi * 3 * (t - channel_delay))
    )

# Add Gaussian noise
noisy_das_data = clean_das_data + np.random.normal(
    0, noise_level, clean_das_data.shape
)

# Denoise along time axis (axis=1)
print("\nDenoising along time axis...")
denoised_das_data = denoise_2d_signal(
    noisy_das_data,
    wavelet='db4',
    threshold_mode='soft',
    threshold_method='universal',
    axis=1  # Denoise along time axis
)

# Calculate metrics
snr_before = calculate_snr(clean_das_data.ravel(), noisy_das_data.ravel())
snr_after = calculate_snr(clean_das_data.ravel(), denoised_das_data.ravel())
rmse_before = calculate_rmse(clean_das_data.ravel(), noisy_das_data.ravel())
rmse_after = calculate_rmse(clean_das_data.ravel(), denoised_das_data.ravel())

# Print results
print("\n" + "="*60)
print("DAS DATA DENOISING RESULTS")
print("="*60)
print(f"Data shape: {n_channels} channels × {n_samples} samples")
print(f"\nBefore denoising:")
print(f"  SNR:  {snr_before:.2f} dB")
print(f"  RMSE: {rmse_before:.4f}")
print(f"\nAfter denoising:")
print(f"  SNR:  {snr_after:.2f} dB")
print(f"  RMSE: {rmse_after:.4f}")
print(f"\nImprovement:")
print(f"  SNR:  +{snr_after - snr_before:.2f} dB")
print(f"  RMSE: -{rmse_before - rmse_after:.4f}")
print("="*60)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Clean DAS data
im1 = axes[0, 0].imshow(
    clean_das_data,
    aspect='auto',
    cmap='seismic',
    extent=[0, 2, n_channels, 0],
    vmin=-2, vmax=2
)
axes[0, 0].set_title('Clean DAS Data')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Channel')
plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')

# Plot 2: Noisy DAS data
im2 = axes[0, 1].imshow(
    noisy_das_data,
    aspect='auto',
    cmap='seismic',
    extent=[0, 2, n_channels, 0],
    vmin=-2, vmax=2
)
axes[0, 1].set_title(f'Noisy DAS Data (SNR: {snr_before:.2f} dB)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Channel')
plt.colorbar(im2, ax=axes[0, 1], label='Amplitude')

# Plot 3: Denoised DAS data
im3 = axes[1, 0].imshow(
    denoised_das_data,
    aspect='auto',
    cmap='seismic',
    extent=[0, 2, n_channels, 0],
    vmin=-2, vmax=2
)
axes[1, 0].set_title(f'Denoised DAS Data (SNR: {snr_after:.2f} dB)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Channel')
plt.colorbar(im3, ax=axes[1, 0], label='Amplitude')

# Plot 4: Noise removed (difference)
noise_removed = noisy_das_data - denoised_das_data
im4 = axes[1, 1].imshow(
    noise_removed,
    aspect='auto',
    cmap='seismic',
    extent=[0, 2, n_channels, 0],
    vmin=-1, vmax=1
)
axes[1, 1].set_title('Removed Noise')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Channel')
plt.colorbar(im4, ax=axes[1, 1], label='Amplitude')

plt.tight_layout()
plt.savefig('examples/das_data_denoising_result.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: examples/das_data_denoising_result.png")

# Plot single channel comparison
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8))

channel_idx = 25  # Middle channel
t_plot = np.linspace(0, 2, n_samples)

axes2[0].plot(t_plot, clean_das_data[channel_idx, :], 'b-', linewidth=1.5)
axes2[0].set_title(f'Channel {channel_idx} - Clean Signal')
axes2[0].set_ylabel('Amplitude')
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t_plot, noisy_das_data[channel_idx, :], 'r-', alpha=0.6, linewidth=0.8)
axes2[1].plot(t_plot, clean_das_data[channel_idx, :], 'b-', alpha=0.3, linewidth=1.5)
axes2[1].set_title(f'Channel {channel_idx} - Noisy Signal')
axes2[1].set_ylabel('Amplitude')
axes2[1].grid(True, alpha=0.3)

axes2[2].plot(t_plot, denoised_das_data[channel_idx, :], 'g-', linewidth=1.5, label='Denoised')
axes2[2].plot(t_plot, clean_das_data[channel_idx, :], 'b--', alpha=0.5, linewidth=1, label='Clean')
axes2[2].set_title(f'Channel {channel_idx} - Denoised Signal')
axes2[2].set_xlabel('Time (s)')
axes2[2].set_ylabel('Amplitude')
axes2[2].legend()
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/das_single_channel_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/das_single_channel_comparison.png")

plt.show()
