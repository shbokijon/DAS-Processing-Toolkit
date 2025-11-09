"""
Demo script for DAS Processing Toolkit visualization functions.

This script demonstrates all three main visualization functions:
- plot_das_section(): Wiggle trace and variable density display
- plot_spectrogram(): Frequency analysis
- plot_velocity_profile(): VSP velocity profiles
"""

import numpy as np
import matplotlib.pyplot as plt
from das_toolkit import plot_das_section, plot_spectrogram, plot_velocity_profile


def generate_synthetic_das_data(n_time=1000, n_channels=50, noise_level=0.1):
    """
    Generate synthetic DAS data for demonstration.

    Parameters
    ----------
    n_time : int
        Number of time samples
    n_channels : int
        Number of channels
    noise_level : float
        Noise amplitude relative to signal

    Returns
    -------
    data : np.ndarray
        Synthetic DAS data (n_time, n_channels)
    time : np.ndarray
        Time axis in seconds
    distance : np.ndarray
        Distance axis in meters
    """
    time = np.linspace(0, 2, n_time)  # 2 seconds
    distance = np.linspace(0, 100, n_channels)  # 100 meters

    # Create synthetic seismic event
    data = np.zeros((n_time, n_channels))

    # Add a linear moveout event (simulating wave propagation)
    velocity = 2000  # m/s
    center_time = 1.0  # s

    for i, dist in enumerate(distance):
        # Travel time based on distance and velocity
        arrival_time = center_time + dist / velocity

        # Create a Ricker wavelet
        t0 = time - arrival_time
        freq = 25  # Hz
        wavelet = (1 - 2 * (np.pi * freq * t0) ** 2) * np.exp(-(np.pi * freq * t0) ** 2)

        # Add to data with amplitude decay
        amplitude = np.exp(-dist / 200)  # Amplitude decay with distance
        data[:, i] = amplitude * wavelet

    # Add noise
    data += noise_level * np.random.randn(n_time, n_channels)

    return data, time, distance


def demo_das_section():
    """Demonstrate plot_das_section() with various options."""
    print("Generating DAS section plots...")

    # Generate synthetic data
    data, time, distance = generate_synthetic_das_data()

    # Example 1: Variable density only
    fig, ax = plot_das_section(
        data, time, distance,
        wiggle=False,
        title='DAS Section - Variable Density',
        cmap='seismic',
        figsize=(12, 8)
    )
    plt.savefig('examples/das_section_vd.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/das_section_vd.png")

    # Example 2: Wiggle traces with variable density
    fig, ax = plot_das_section(
        data, time, distance,
        wiggle=True,
        wiggle_scale=1.5,
        wiggle_skip=2,
        title='DAS Section - Wiggle + Variable Density',
        cmap='RdBu_r',
        figsize=(12, 8)
    )
    plt.savefig('examples/das_section_wiggle.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/das_section_wiggle.png")

    # Example 3: Wiggle traces only (using grayscale background)
    fig, ax = plot_das_section(
        data, time, distance,
        wiggle=True,
        wiggle_scale=2.0,
        wiggle_fill=True,
        wiggle_skip=1,
        title='DAS Section - Wiggle Traces',
        cmap='Greys',
        vmin=-0.3,
        vmax=0.3,
        figsize=(12, 8)
    )
    plt.savefig('examples/das_section_wiggle_only.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/das_section_wiggle_only.png")


def demo_spectrogram():
    """Demonstrate plot_spectrogram() with various signals."""
    print("\nGenerating spectrogram plots...")

    # Generate synthetic signal with varying frequency content
    fs = 1000  # Sampling frequency (Hz)
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Create a chirp signal (frequency increases over time) + constant tone
    f0, f1 = 10, 200  # Frequency range for chirp
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2))

    # Add a constant 50 Hz tone
    constant_tone = 0.5 * np.sin(2 * np.pi * 50 * t)

    # Add some noise
    noise = 0.1 * np.random.randn(len(t))

    signal = chirp + constant_tone + noise

    # Example 1: Standard spectrogram
    fig, ax, f, t_spec, Sxx = plot_spectrogram(
        signal,
        fs=fs,
        title='Spectrogram - Chirp + 50 Hz Tone',
        fmax=250,
        nperseg=512,
        figsize=(12, 6)
    )
    plt.savefig('examples/spectrogram_standard.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/spectrogram_standard.png")

    # Example 2: High-resolution spectrogram
    fig, ax, f, t_spec, Sxx = plot_spectrogram(
        signal,
        fs=fs,
        title='Spectrogram - High Resolution',
        fmax=250,
        nperseg=1024,
        noverlap=896,
        window='hamming',
        cmap='inferno',
        figsize=(12, 6)
    )
    plt.savefig('examples/spectrogram_highres.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/spectrogram_highres.png")

    # Example 3: Linear scale spectrogram
    fig, ax, f, t_spec, Sxx = plot_spectrogram(
        signal,
        fs=fs,
        title='Spectrogram - Linear Scale',
        fmax=250,
        scale='linear',
        cmap='plasma',
        figsize=(12, 6)
    )
    plt.savefig('examples/spectrogram_linear.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/spectrogram_linear.png")


def demo_velocity_profile():
    """Demonstrate plot_velocity_profile() for VSP analysis."""
    print("\nGenerating velocity profile plots...")

    # Generate synthetic VSP data
    depth = np.linspace(0, 1000, 50)  # meters

    # Create realistic velocity profile with layers
    velocity = np.zeros_like(depth)

    # Layer 1: 0-300m (unconsolidated sediments)
    mask1 = depth < 300
    velocity[mask1] = 1500 + 1.0 * depth[mask1]

    # Layer 2: 300-600m (consolidated sediments)
    mask2 = (depth >= 300) & (depth < 600)
    velocity[mask2] = 1800 + 1.5 * (depth[mask2] - 300)

    # Layer 3: 600-1000m (bedrock)
    mask3 = depth >= 600
    velocity[mask3] = 2250 + 2.0 * (depth[mask3] - 600)

    # Add some realistic variations
    velocity += 50 * np.sin(depth / 100)

    # Add uncertainty
    velocity_error = 30 + 0.05 * velocity

    # Example 1: Simple velocity profile with error bars
    fig, ax = plot_velocity_profile(
        depth, velocity, velocity_error,
        title='VSP Velocity Profile',
        figsize=(8, 10)
    )
    plt.savefig('examples/velocity_profile_simple.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/velocity_profile_simple.png")

    # Example 2: Multiple velocity models comparison
    # Create alternative models
    other_profiles = {
        'Gardner Model': (depth, 1500 + 0.8 * depth),
        'Linear Model': (depth, 1600 + 1.2 * depth),
    }

    fig, ax = plot_velocity_profile(
        depth, velocity, velocity_error,
        velocity_profiles=other_profiles,
        title='VSP Velocity Profile - Model Comparison',
        figsize=(8, 10)
    )
    plt.savefig('examples/velocity_profile_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/velocity_profile_comparison.png")

    # Example 3: With geological layer markers
    markers = {
        'Layer 1': 300,
        'Layer 2': 600,
    }
    marker_labels = {
        'Layer 1': 'Consolidated Sediments',
        'Layer 2': 'Bedrock',
    }

    fig, ax = plot_velocity_profile(
        depth, velocity, velocity_error,
        velocity_profiles=other_profiles,
        markers=markers,
        marker_labels=marker_labels,
        title='VSP Velocity Profile - With Geological Markers',
        figsize=(9, 10)
    )
    plt.savefig('examples/velocity_profile_markers.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/velocity_profile_markers.png")


def demo_combined_figure():
    """Create a combined figure with all three plot types."""
    print("\nGenerating combined figure...")

    # Generate data
    das_data, time, distance = generate_synthetic_das_data()

    # Generate signal for spectrogram
    fs = 500
    t = np.linspace(0, 2, fs * 2)
    signal = np.sin(2 * np.pi * 25 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    signal += 0.1 * np.random.randn(len(t))

    # Generate VSP data
    depth = np.linspace(0, 1000, 30)
    velocity = 1500 + 0.8 * depth + 30 * np.sin(depth / 100)
    velocity_error = 50 * np.ones_like(velocity)

    # Create combined figure
    fig = plt.figure(figsize=(16, 12))

    # DAS section
    ax1 = plt.subplot(2, 2, (1, 2))
    plot_das_section(
        das_data, time, distance,
        wiggle=True,
        wiggle_skip=2,
        title='(a) DAS Section',
        ax=ax1
    )

    # Spectrogram
    ax2 = plt.subplot(2, 2, 3)
    plot_spectrogram(
        signal, fs=fs,
        title='(b) Spectrogram',
        fmax=100,
        ax=ax2
    )

    # Velocity profile
    ax3 = plt.subplot(2, 2, 4)
    plot_velocity_profile(
        depth, velocity, velocity_error,
        title='(c) Velocity Profile',
        ax=ax3,
        figsize=(6, 8)  # This will be ignored since ax is provided
    )

    plt.tight_layout()
    plt.savefig('examples/combined_figure.png', dpi=150, bbox_inches='tight')
    print("  Saved: examples/combined_figure.png")


if __name__ == '__main__':
    print("DAS Processing Toolkit - Visualization Demo")
    print("=" * 50)

    # Run all demos
    demo_das_section()
    demo_spectrogram()
    demo_velocity_profile()
    demo_combined_figure()

    print("\n" + "=" * 50)
    print("All demos completed successfully!")
    print("Check the 'examples/' directory for output images.")
