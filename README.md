# DAS-Processing-Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()

A Python toolkit for digital signal processing and preprocessing of Distributed Acoustic Sensing (DAS) data.

## Overview

Distributed Acoustic Sensing (DAS) is an advanced technology that transforms fiber optic cables into distributed sensor arrays, enabling continuous measurement of acoustic signals, vibrations, and strain along the entire cable length. DAS systems generate large volumes of high-dimensional spatiotemporal data that require sophisticated signal processing techniques for analysis and interpretation.

The **DAS-Processing-Toolkit** provides a comprehensive suite of digital filtering algorithms specifically designed to preprocess DAS data. The toolkit addresses common challenges in DAS data analysis including:

- **Noise reduction**: Removal of environmental and instrumental noise from DAS signals
- **Signal enhancement**: Improvement of signal-to-noise ratio for weak signals
- **Frequency filtering**: Isolation of signals in specific frequency bands
- **Data conditioning**: Preparation of DAS data for downstream analysis and machine learning applications

### Motivation

DAS technology is increasingly deployed across diverse applications including:

- Seismic monitoring and earthquake detection
- Infrastructure monitoring (bridges, pipelines, railways)
- Borehole geophysical measurements
- Perimeter security and intrusion detection
- Traffic monitoring and smart cities
- Subsurface imaging and exploration

Each application domain requires tailored signal processing workflows to extract meaningful information from raw DAS data. This toolkit provides researchers and practitioners with robust, well-tested filtering algorithms to streamline their DAS data processing pipelines.

## Features

- **Digital Filter Implementations**: Butterworth, Chebyshev, Elliptic, and Bessel filters
- **Frequency Domain Processing**: FFT-based filtering and spectral analysis
- **Temporal Filtering**: Bandpass, lowpass, highpass, and bandstop filters
- **Spatial Filtering**: Along-fiber filtering for coherent noise removal
- **Batch Processing**: Efficient processing of large DAS datasets
- **NumPy Integration**: Seamless integration with scientific Python ecosystem

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy >= 1.19.0
- SciPy >= 1.5.0

### Install via pip

```bash
# Installation from PyPI (once published)
pip install das-processing-toolkit

# Installation from source
git clone https://github.com/shbokijon/DAS-Processing-Toolkit.git
cd DAS-Processing-Toolkit
pip install -e .
```

### Install via conda

```bash
# Create a new conda environment (recommended)
conda create -n das-processing python=3.9
conda activate das-processing

# Install the package (once published to conda-forge)
conda install -c conda-forge das-processing-toolkit

# Or install from source
git clone https://github.com/shbokijon/DAS-Processing-Toolkit.git
cd DAS-Processing-Toolkit
pip install -e .
```

## Quick Start

Here's a simple example demonstrating how to apply a Butterworth bandpass filter to DAS data:

```python
import numpy as np
from das_processing import filters

# Load your DAS data (shape: [n_channels, n_samples])
# Each row represents one channel along the fiber
das_data = np.load('path/to/your/das_data.npy')
sampling_rate = 1000  # Hz

# Design a Butterworth bandpass filter (10-100 Hz)
filtered_data = filters.butterworth_bandpass(
    data=das_data,
    lowcut=10.0,      # Lower frequency bound (Hz)
    highcut=100.0,    # Upper frequency bound (Hz)
    fs=sampling_rate, # Sampling frequency (Hz)
    order=4           # Filter order
)

# Apply additional noise reduction
cleaned_data = filters.median_filter_spatial(
    data=filtered_data,
    kernel_size=3  # Filter along spatial (channel) dimension
)

# Save processed data
np.save('processed_das_data.npy', cleaned_data)
```

### Working with Time-Series Data

```python
from das_processing import filters, utils

# Load DAS data
das_data, metadata = utils.load_das_file('recording.h5')

# Extract a single channel for processing
channel_idx = 100
signal = das_data[channel_idx, :]

# Apply lowpass filter to remove high-frequency noise
signal_filtered = filters.butterworth_lowpass(
    data=signal,
    cutoff=50.0,      # Cutoff frequency (Hz)
    fs=metadata['sampling_rate'],
    order=5
)

# Visualize results
import matplotlib.pyplot as plt
utils.plot_comparison(signal, signal_filtered, fs=metadata['sampling_rate'])
plt.savefig('filtering_result.png')
```

## Documentation

Full API documentation is available at: [https://das-processing-toolkit.readthedocs.io](https://das-processing-toolkit.readthedocs.io) *(in development)*

For additional examples and tutorials, see the [`examples/`](examples/) directory.

## Project Status

This project is currently in active development. Core filtering functionality is being implemented. Contributions, bug reports, and feature requests are welcome!

### Roadmap

- [x] Project initialization and repository setup
- [ ] Butterworth filter implementation
- [ ] Chebyshev filter implementation
- [ ] Spectral filtering methods
- [ ] Spatial filtering algorithms
- [ ] Comprehensive test suite
- [ ] Documentation and tutorials
- [ ] Performance optimization
- [ ] PyPI and conda-forge distribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use DAS-Processing-Toolkit in your research, please cite:

```bibtex
@software{das_processing_toolkit,
  author       = {shbokijon},
  title        = {DAS-Processing-Toolkit: Digital Signal Processing for Distributed Acoustic Sensing Data},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/shbokijon/DAS-Processing-Toolkit}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- Inspired by best practices in scientific Python development

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This toolkit is designed for research and educational purposes. For production deployments, please thoroughly validate the filtering parameters for your specific DAS system and application.
