# DAS Visualization Examples

This directory contains example scripts demonstrating the visualization capabilities of the DAS Processing Toolkit.

## Running the Demo

To run the complete demonstration:

```bash
python demo_visualization.py
```

This will generate several example plots showing:

1. **DAS Section Plots** - Various styles of wiggle trace and variable density displays
2. **Spectrograms** - Frequency analysis with different parameters
3. **Velocity Profiles** - VSP analysis with error bands and geological markers
4. **Combined Figure** - All three plot types in a single publication-quality figure

## Output

The script will create several PNG images in the `examples/` directory:

- `das_section_vd.png` - Variable density display only
- `das_section_wiggle.png` - Wiggle traces with variable density
- `das_section_wiggle_only.png` - Wiggle traces emphasized
- `spectrogram_standard.png` - Standard spectrogram
- `spectrogram_highres.png` - High-resolution spectrogram
- `spectrogram_linear.png` - Linear scale spectrogram
- `velocity_profile_simple.png` - Basic velocity profile with error bars
- `velocity_profile_comparison.png` - Multiple velocity models
- `velocity_profile_markers.png` - With geological layer markers
- `combined_figure.png` - All plot types combined

## Requirements

Make sure you have installed the required packages:

```bash
pip install -r ../requirements.txt
```

Or install the package in development mode:

```bash
pip install -e ..
```
