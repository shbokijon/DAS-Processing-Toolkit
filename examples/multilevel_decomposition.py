"""
Multi-level Wavelet Decomposition Example

This example demonstrates multi-level wavelet decomposition and shows
how different decomposition levels affect the denoising results.
"""

import numpy as np
import matplotlib.pyplot as plt
from das_toolkit import multilevel_wavelet_denoise, calculate_snr, calculate_rmse

# Set random seed for reproducibility
np.random.seed(42)

# Create a complex signal with multiple frequency components
print("Generating multi-frequency signal...")
t = np.linspace(0, 1, 2000)

# Low frequency component
low_freq = np.sin(2 * np.pi * 3 * t)

# Medium frequency component
med_freq = 0.5 * np.sin(2 * np.pi * 15 * t)

# High frequency component
high_freq = 0.3 * np.sin(2 * np.pi * 50 * t)

# Combine frequencies
clean_signal = low_freq + med_freq + high_freq

# Add noise
noise_level = 0.4
noisy_signal = clean_signal + np.random.normal(0, noise_level, len(clean_signal))

# Perform multi-level denoising
print("Performing multi-level wavelet decomposition...")
max_level = 6
results = multilevel_wavelet_denoise(
    noisy_signal,
    wavelet='db4',
    max_level=max_level,
    threshold_mode='soft',
    threshold_method='universal',
    return_all_levels=True
)

# Calculate metrics for each level
print("\n" + "="*70)
print("MULTI-LEVEL DENOISING RESULTS")
print("="*70)
print(f"{'Level':<8} {'SNR (dB)':<12} {'RMSE':<12} {'SNR Gain (dB)':<15}")
print("-"*70)

snr_noisy = calculate_snr(clean_signal, noisy_signal)
rmse_noisy = calculate_rmse(clean_signal, noisy_signal)

print(f"{'Noisy':<8} {snr_noisy:<12.2f} {rmse_noisy:<12.4f} {'-':<15}")
print("-"*70)

metrics = []
for level in sorted(results.keys()):
    denoised = results[level]
    snr = calculate_snr(clean_signal, denoised)
    rmse = calculate_rmse(clean_signal, denoised)
    snr_gain = snr - snr_noisy

    metrics.append({
        'level': level,
        'snr': snr,
        'rmse': rmse,
        'snr_gain': snr_gain
    })

    print(f"{level:<8} {snr:<12.2f} {rmse:<12.4f} {snr_gain:<15.2f}")

print("="*70)

# Find best level
best_level = max(metrics, key=lambda x: x['snr'])['level']
print(f"\nBest level: {best_level} (SNR: {best_level} dB)")

# Visualize results for selected levels
selected_levels = [1, 2, 4, 6]
fig, axes = plt.subplots(len(selected_levels) + 1, 1, figsize=(14, 12))

# Plot noisy signal
axes[0].plot(t, clean_signal, 'b-', linewidth=1, alpha=0.5, label='Clean')
axes[0].plot(t, noisy_signal, 'r-', linewidth=0.8, alpha=0.7, label='Noisy')
axes[0].set_title(f'Noisy Signal (SNR: {snr_noisy:.2f} dB)')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot denoised signals for each selected level
for idx, level in enumerate(selected_levels, start=1):
    denoised = results[level]
    metric = [m for m in metrics if m['level'] == level][0]

    axes[idx].plot(t, clean_signal, 'b-', linewidth=1, alpha=0.5, label='Clean')
    axes[idx].plot(t, denoised, 'g-', linewidth=1.2, alpha=0.8, label='Denoised')

    axes[idx].set_title(
        f'Level {level} Denoising (SNR: {metric["snr"]:.2f} dB, '
        f'Gain: +{metric["snr_gain"]:.2f} dB)'
    )
    axes[idx].set_ylabel('Amplitude')
    axes[idx].legend(loc='upper right')
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('examples/multilevel_decomposition.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/multilevel_decomposition.png")

# Plot SNR vs decomposition level
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

levels = [m['level'] for m in metrics]
snrs = [m['snr'] for m in metrics]
rmses = [m['rmse'] for m in metrics]
snr_gains = [m['snr_gain'] for m in metrics]

# SNR plot
ax1.plot(levels, snrs, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(snr_noisy, color='r', linestyle='--', label='Noisy signal')
ax1.axvline(best_level, color='g', linestyle=':', alpha=0.5, label=f'Best level ({best_level})')
ax1.set_xlabel('Decomposition Level')
ax1.set_ylabel('SNR (dB)')
ax1.set_title('SNR vs Decomposition Level')
ax1.grid(True, alpha=0.3)
ax1.legend()

# RMSE plot
ax2.plot(levels, rmses, 'o-', linewidth=2, markersize=8, color='coral')
ax2.axhline(rmse_noisy, color='r', linestyle='--', label='Noisy signal')
ax2.axvline(best_level, color='g', linestyle=':', alpha=0.5, label=f'Best level ({best_level})')
ax2.set_xlabel('Decomposition Level')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE vs Decomposition Level')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('examples/multilevel_metrics.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/multilevel_metrics.png")

# Show frequency content analysis
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9))

from scipy import signal as scipy_signal

# Compute and plot spectrograms
def plot_spectrogram(ax, sig, title):
    f, t_spec, Sxx = scipy_signal.spectrogram(sig, fs=len(sig), nperseg=256)
    ax.pcolormesh(t_spec, f[:200], Sxx[:200], shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

plot_spectrogram(axes3[0], clean_signal, 'Clean Signal - Spectrogram')
plot_spectrogram(axes3[1], noisy_signal, 'Noisy Signal - Spectrogram')
plot_spectrogram(axes3[2], results[best_level], f'Denoised (Level {best_level}) - Spectrogram')
axes3[2].set_xlabel('Time')

plt.tight_layout()
plt.savefig('examples/multilevel_spectrogram.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/multilevel_spectrogram.png")

print("\n" + "="*70)
print("Analysis complete! Generated visualizations:")
print("  1. examples/multilevel_decomposition.png")
print("  2. examples/multilevel_metrics.png")
print("  3. examples/multilevel_spectrogram.png")
print("="*70)

plt.show()
