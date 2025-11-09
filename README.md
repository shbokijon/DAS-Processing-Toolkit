# DAS Processing Toolkit

Digital filters for DAS (Distributed Acoustic Sensing) data preprocessing with advanced wavelet denoising capabilities.

## Features

- **Wavelet-based Denoising**: State-of-the-art wavelet transform denoising using PyWavelets
- **Multiple Thresholding Methods**:
  - Soft thresholding (shrinkage)
  - Hard thresholding (keep/remove)
- **Threshold Calculation Methods**:
  - Universal threshold (Donoho & Johnstone)
  - SURE (Stein's Unbiased Risk Estimate)
  - Minimax threshold
- **Multi-level Decomposition**: Analyze and denoise at different wavelet decomposition levels
- **2D Signal Processing**: Specialized support for DAS data matrices (channels × time)
- **Comparison Tools**: Compare different denoising methods with comprehensive visualizations
- **Comprehensive Examples**: Ready-to-run examples for various use cases

## Installation

### From source

```bash
git clone https://github.com/shbokijon/DAS-Processing-Toolkit.git
cd DAS-Processing-Toolkit
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- PyWavelets >= 1.1.1
- Matplotlib >= 3.3.0
- SciPy >= 1.6.0

## Quick Start

### Basic Denoising

```python
import numpy as np
from das_toolkit import wavelet_denoise

# Create noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)
noisy_signal = clean_signal + np.random.normal(0, 0.2, len(clean_signal))

# Denoise using wavelet transform
denoised = wavelet_denoise(
    noisy_signal,
    wavelet='db4',           # Daubechies 4 wavelet
    threshold_mode='soft',   # Soft thresholding
    threshold_method='universal'
)
```

### Compare Different Methods

```python
from das_toolkit import compare_denoising_methods, visualize_denoising_comparison

# Compare different wavelet families and thresholding modes
results = compare_denoising_methods(
    noisy_signal,
    wavelets=['db4', 'sym4', 'coif3', 'haar'],
    threshold_modes=['soft', 'hard'],
    threshold_methods=['universal']
)

# Visualize comparison
fig = visualize_denoising_comparison(
    clean_signal,
    noisy_signal,
    results,
    title="Wavelet Denoising Comparison"
)
```

### Multi-level Decomposition

```python
from das_toolkit import multilevel_wavelet_denoise

# Get denoised signals at all decomposition levels
results = multilevel_wavelet_denoise(
    noisy_signal,
    wavelet='db4',
    max_level=6,
    return_all_levels=True
)

# results[1], results[2], ..., results[6] contain denoised signals
```

### DAS Data (2D) Denoising

```python
from das_toolkit.denoising import denoise_2d_signal

# DAS data: channels × time samples
das_data = np.random.randn(100, 10000)  # 100 channels, 10000 samples

# Denoise along time axis
denoised_das = denoise_2d_signal(
    das_data,
    wavelet='db4',
    axis=1  # Denoise along time axis
)
```

## Examples

The `examples/` directory contains comprehensive examples:

### 1. Basic Denoising (`basic_denoising.py`)
Demonstrates basic wavelet denoising on a synthetic signal with visualization.

```bash
python examples/basic_denoising.py
```

### 2. Method Comparison (`compare_methods.py`)
Compares different wavelet families, thresholding modes, and threshold methods.

```bash
python examples/compare_methods.py
```

### 3. DAS Data Processing (`das_data_denoising.py`)
Shows how to denoise 2D DAS data with multiple channels.

```bash
python examples/das_data_denoising.py
```

### 4. Multi-level Decomposition (`multilevel_decomposition.py`)
Analyzes the effect of different decomposition levels on denoising quality.

```bash
python examples/multilevel_decomposition.py
```

## API Reference

### Main Functions

#### `wavelet_denoise(signal, wavelet='db4', level=None, threshold_method='universal', threshold_mode='soft', noise_std=None, mode='symmetric')`

Denoise a 1D signal using wavelet transform with thresholding.

**Parameters:**
- `signal` (np.ndarray): Input signal (1D array)
- `wavelet` (str): Wavelet type (e.g., 'db4', 'sym4', 'coif3', 'haar')
- `level` (int, optional): Decomposition level. If None, uses maximum useful level
- `threshold_method` (str): Method for calculating threshold ('universal', 'sure', 'minimax')
- `threshold_mode` (str): Thresholding mode ('soft' or 'hard')
- `noise_std` (float, optional): Known noise standard deviation
- `mode` (str): Signal extension mode for wavelet transform

**Returns:**
- `np.ndarray`: Denoised signal

#### `multilevel_wavelet_denoise(signal, wavelet='db4', max_level=None, threshold_method='universal', threshold_mode='soft', mode='symmetric', return_all_levels=False)`

Perform multi-level wavelet decomposition and denoising.

**Parameters:**
- `signal` (np.ndarray): Input signal (1D array)
- `wavelet` (str): Wavelet type
- `max_level` (int, optional): Maximum decomposition level
- `threshold_method` (str): Threshold calculation method
- `threshold_mode` (str): 'soft' or 'hard' thresholding
- `mode` (str): Signal extension mode
- `return_all_levels` (bool): If True, returns dict with results for each level

**Returns:**
- `np.ndarray` or `dict`: Denoised signal or dictionary of results for each level

#### `compare_denoising_methods(signal, wavelets=None, threshold_modes=None, threshold_methods=None, level=None)`

Compare different denoising methods on the same signal.

**Parameters:**
- `signal` (np.ndarray): Input noisy signal
- `wavelets` (list, optional): List of wavelet types to compare
- `threshold_modes` (list, optional): List of threshold modes to compare
- `threshold_methods` (list, optional): List of threshold methods to compare
- `level` (int, optional): Decomposition level

**Returns:**
- `dict`: Dictionary mapping method name to denoised signal

#### `visualize_denoising_comparison(original, noisy, denoised_signals, title='Wavelet Denoising Comparison', figsize=(15, 10), save_path=None)`

Visualize and compare different denoising results.

**Parameters:**
- `original` (np.ndarray): Original clean signal
- `noisy` (np.ndarray): Noisy signal
- `denoised_signals` (dict): Dictionary mapping method name to denoised signal
- `title` (str): Figure title
- `figsize` (tuple): Figure size
- `save_path` (str, optional): Path to save figure

**Returns:**
- `matplotlib.figure.Figure`: The created figure

### Utility Functions

#### `calculate_snr(clean_signal, noisy_signal)`
Calculate Signal-to-Noise Ratio in decibels.

#### `calculate_rmse(clean_signal, noisy_signal)`
Calculate Root Mean Square Error.

#### `estimate_noise_std(coeffs, method='mad')`
Estimate noise standard deviation from wavelet coefficients.

#### `calculate_threshold(coeffs, method='universal', noise_std=None)`
Calculate threshold value for wavelet coefficient shrinkage.

## Wavelet Families

Supported wavelet families include:
- **Daubechies**: `db1`, `db2`, `db3`, `db4`, ..., `db20`
- **Symlets**: `sym2`, `sym3`, `sym4`, ..., `sym20`
- **Coiflets**: `coif1`, `coif2`, `coif3`, ..., `coif17`
- **Biorthogonal**: `bior1.1`, `bior1.3`, `bior1.5`, ...
- **Haar**: `haar`
- And many more from PyWavelets

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=das_toolkit --cov-report=html
```

## Theory

### Wavelet Denoising

Wavelet denoising works by:

1. **Decomposition**: Transform the signal into wavelet domain using Discrete Wavelet Transform (DWT)
2. **Thresholding**: Apply thresholding to wavelet coefficients to remove noise
3. **Reconstruction**: Transform back to time domain using Inverse DWT

### Thresholding Methods

- **Soft Thresholding**: Shrinks coefficients towards zero
  - If |x| < threshold: x → 0
  - If |x| ≥ threshold: x → sign(x) × (|x| - threshold)

- **Hard Thresholding**: Keeps or removes coefficients
  - If |x| < threshold: x → 0
  - If |x| ≥ threshold: x → x (unchanged)

### Threshold Selection

- **Universal**: σ × √(2 log N), where σ is noise std and N is signal length
- **SURE**: Stein's Unbiased Risk Estimate - minimizes MSE
- **Minimax**: Minimax principle for worst-case scenarios

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Donoho, D. L., & Johnstone, J. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.
2. Mallat, S. (2008). *A Wavelet Tour of Signal Processing: The Sparse Way* (3rd ed.). Academic Press.
3. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{das_toolkit,
  title = {DAS Processing Toolkit},
  author = {DAS Processing Toolkit Contributors},
  year = {2025},
  url = {https://github.com/shbokijon/DAS-Processing-Toolkit}
}
```