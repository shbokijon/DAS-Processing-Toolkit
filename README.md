# DAS Processing Toolkit

A comprehensive Python toolkit for processing and visualizing Distributed Acoustic Sensing (DAS) data with publication-quality figures.

## Features

### Visualization Module

The toolkit provides three main visualization functions for DAS data analysis:

1. **`plot_das_section()`** - Wiggle trace and variable density displays
   - Classic seismic section visualization
   - Customizable wiggle traces with fill
   - Variable density (image) display
   - Publication-quality matplotlib figures

2. **`plot_spectrogram()`** - Frequency analysis
   - Short-Time Fourier Transform (STFT) based spectrograms
   - Customizable time-frequency resolution
   - Multiple window functions and colormaps
   - dB or linear scale options

3. **`plot_velocity_profile()`** - VSP (Vertical Seismic Profiling) results
   - Velocity vs depth plotting
   - Error bars and uncertainty bands
   - Multiple profile comparison
   - Geological layer markers

## Installation

### From source

```bash
git clone https://github.com/shbokijon/DAS-Processing-Toolkit.git
cd DAS-Processing-Toolkit
pip install -e .
```

### Requirements

- Python >= 3.7
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## Quick Start

### 1. DAS Section Visualization

```python
import numpy as np
from das_toolkit import plot_das_section

# Load or generate DAS data
data = np.random.randn(1000, 50)  # (time_samples, channels)
time = np.linspace(0, 2, 1000)    # seconds
distance = np.linspace(0, 100, 50)  # meters

# Create wiggle trace plot
fig, ax = plot_das_section(
    data, time, distance,
    wiggle=True,
    wiggle_scale=1.5,
    wiggle_skip=2,
    title='DAS Section',
    cmap='seismic'
)
```

### 2. Spectrogram Analysis

```python
from das_toolkit import plot_spectrogram

# Generate or load signal
fs = 1000  # Sampling frequency in Hz
signal = np.sin(2 * np.pi * 50 * np.linspace(0, 10, fs * 10))

# Plot spectrogram
fig, ax, f, t, Sxx = plot_spectrogram(
    signal,
    fs=fs,
    fmax=200,
    nperseg=512,
    title='Frequency Analysis',
    cmap='viridis'
)
```

### 3. Velocity Profile (VSP)

```python
from das_toolkit import plot_velocity_profile

# VSP data
depth = np.linspace(0, 1000, 50)  # meters
velocity = 1500 + 0.6 * depth      # m/s
velocity_error = 50 * np.ones_like(velocity)

# Plot velocity profile
fig, ax = plot_velocity_profile(
    depth, velocity,
    velocity_error=velocity_error,
    title='VSP Velocity Profile'
)
```

## Advanced Usage

### Multiple Velocity Models Comparison

```python
# Define alternative velocity models
other_profiles = {
    'Gardner Model': (depth, 1500 + 0.8 * depth),
    'Linear Model': (depth, 1600 + 1.2 * depth),
}

# Add geological markers
markers = {
    'Layer 1': 300,
    'Layer 2': 600,
}

fig, ax = plot_velocity_profile(
    depth, velocity, velocity_error,
    velocity_profiles=other_profiles,
    markers=markers,
    title='Velocity Profile Comparison'
)
```

### High-Resolution Spectrogram

```python
fig, ax, f, t, Sxx = plot_spectrogram(
    signal,
    fs=1000,
    nperseg=1024,      # Longer segments for better frequency resolution
    noverlap=896,      # High overlap for better time resolution
    window='hamming',
    fmin=10,
    fmax=200,
    scale='dB',
    cmap='inferno'
)
```

### Custom DAS Section Styling

```python
fig, ax = plot_das_section(
    data, time, distance,
    wiggle=True,
    wiggle_scale=2.0,
    wiggle_color='darkblue',
    wiggle_fill=True,
    wiggle_fill_color='navy',
    wiggle_skip=1,
    cmap='RdBu_r',
    vmin=-0.5,
    vmax=0.5,
    colorbar_label='Strain Rate',
    grid=True,
    figsize=(14, 10),
    dpi=150
)
```

## Examples

The `examples/` directory contains comprehensive demonstrations of all visualization functions:

```bash
cd examples
python demo_visualization.py
```

This will generate multiple example figures showing:
- Various DAS section display styles
- Different spectrogram configurations
- Velocity profile variations
- Combined multi-panel figures

## API Reference

### plot_das_section()

**Parameters:**
- `data` (np.ndarray): 2D array of DAS data (n_time, n_channels)
- `time` (np.ndarray, optional): Time axis values
- `distance` (np.ndarray, optional): Distance/channel axis values
- `wiggle` (bool): Enable wiggle trace overlay
- `wiggle_scale` (float): Scaling factor for wiggle amplitudes
- `wiggle_skip` (int): Plot every Nth trace
- `cmap` (str): Colormap for variable density display
- `figsize` (tuple): Figure size in inches
- `dpi` (int): Resolution in dots per inch
- And many more customization options...

**Returns:**
- `fig`: matplotlib Figure object
- `ax`: matplotlib Axes object

### plot_spectrogram()

**Parameters:**
- `data` (np.ndarray): 1D or 2D signal data
- `fs` (float): Sampling frequency in Hz
- `nperseg` (int): Length of each segment for STFT
- `noverlap` (int): Number of overlapping points
- `window` (str): Window function ('hann', 'hamming', etc.)
- `fmin`, `fmax` (float): Frequency range to display
- `scale` (str): 'dB' or 'linear'
- `cmap` (str): Colormap name

**Returns:**
- `fig`: matplotlib Figure object
- `ax`: matplotlib Axes object
- `f`: Frequency array
- `t`: Time array
- `Sxx`: Spectrogram values

### plot_velocity_profile()

**Parameters:**
- `depth` (np.ndarray): Depth values
- `velocity` (np.ndarray): Velocity values
- `velocity_error` (np.ndarray, optional): Uncertainty values
- `velocity_profiles` (dict, optional): Additional profiles for comparison
- `markers` (dict, optional): Geological layer markers
- `invert_yaxis` (bool): Depth increases downward
- `grid` (bool): Show grid lines

**Returns:**
- `fig`: matplotlib Figure object
- `ax`: matplotlib Axes object

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{das_processing_toolkit,
  title = {DAS Processing Toolkit},
  author = {DAS Processing Toolkit Contributors},
  year = {2024},
  url = {https://github.com/shbokijon/DAS-Processing-Toolkit}
}
```

## Acknowledgments

This toolkit is designed for researchers and engineers working with Distributed Acoustic Sensing data in applications such as:
- Seismic monitoring
- Vertical Seismic Profiling (VSP)
- Structural health monitoring
- Pipeline monitoring
- Traffic monitoring
