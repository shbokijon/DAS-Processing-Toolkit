"""
Visualization module for DAS data processing.

This module provides publication-quality plotting functions for DAS data analysis,
including wiggle trace displays, spectrograms, and velocity profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union, List, Dict, Any
from scipy import signal


def plot_das_section(
    data: np.ndarray,
    time: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None,
    wiggle: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'seismic',
    aspect: str = 'auto',
    wiggle_scale: float = 1.0,
    wiggle_color: str = 'black',
    wiggle_linewidth: float = 0.5,
    wiggle_fill: bool = True,
    wiggle_fill_color: str = 'black',
    wiggle_skip: int = 1,
    xlabel: str = 'Distance (m)',
    ylabel: str = 'Time (s)',
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = 'Amplitude',
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 100,
    grid: bool = False,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot DAS section with wiggle traces and/or variable density display.

    This function creates publication-quality visualizations of DAS data sections,
    commonly used for seismic data display. It supports both variable density
    (image) and wiggle trace representations.

    Parameters
    ----------
    data : np.ndarray
        2D array of DAS data with shape (n_time, n_channels) or (n_channels, n_time).
        The function will automatically detect the orientation based on aspect ratio.
    time : np.ndarray, optional
        Time axis values. If None, uses sample indices.
    distance : np.ndarray, optional
        Distance/channel axis values. If None, uses channel indices.
    wiggle : bool, default=True
        Whether to overlay wiggle traces on the image.
    vmin : float, optional
        Minimum value for color scale. If None, uses data percentile (2%).
    vmax : float, optional
        Maximum value for color scale. If None, uses data percentile (98%).
    cmap : str, default='seismic'
        Colormap name for variable density display.
    aspect : str, default='auto'
        Aspect ratio of the plot ('auto', 'equal', or numeric value).
    wiggle_scale : float, default=1.0
        Scaling factor for wiggle trace amplitudes.
    wiggle_color : str, default='black'
        Color for wiggle traces.
    wiggle_linewidth : float, default=0.5
        Line width for wiggle traces.
    wiggle_fill : bool, default=True
        Whether to fill positive amplitudes of wiggle traces.
    wiggle_fill_color : str, default='black'
        Fill color for wiggle traces.
    wiggle_skip : int, default=1
        Plot every Nth wiggle trace to avoid overcrowding.
    xlabel : str, default='Distance (m)'
        Label for x-axis.
    ylabel : str, default='Time (s)'
        Label for y-axis.
    title : str, optional
        Plot title.
    colorbar : bool, default=True
        Whether to add a colorbar.
    colorbar_label : str, default='Amplitude'
        Label for the colorbar.
    figsize : tuple, default=(12, 8)
        Figure size in inches (width, height).
    dpi : int, default=100
        Dots per inch for the figure.
    grid : bool, default=False
        Whether to show grid lines.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to imshow().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import numpy as np
    >>> from das_toolkit import plot_das_section
    >>>
    >>> # Generate synthetic DAS data
    >>> time = np.linspace(0, 1, 1000)
    >>> distance = np.linspace(0, 100, 50)
    >>> data = np.random.randn(len(time), len(distance))
    >>>
    >>> # Plot with default settings
    >>> fig, ax = plot_das_section(data, time, distance)
    >>>
    >>> # Plot with custom wiggle parameters
    >>> fig, ax = plot_das_section(
    ...     data, time, distance,
    ...     wiggle=True,
    ...     wiggle_scale=2.0,
    ...     wiggle_skip=2,
    ...     cmap='RdBu_r'
    ... )
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D array, got shape {data.shape}")

    # Determine data orientation (time should be the longer axis typically)
    if data.shape[0] < data.shape[1]:
        data = data.T
        swapped = True
    else:
        swapped = False

    n_time, n_channels = data.shape

    # Set up axes
    if time is None:
        time = np.arange(n_time)
    if distance is None:
        distance = np.arange(n_channels)

    # Calculate color limits if not provided
    if vmin is None:
        vmin = np.percentile(data, 2)
    if vmax is None:
        vmax = np.percentile(data, 98)

    # Plot variable density image
    extent = [distance[0], distance[-1], time[-1], time[0]]
    im = ax.imshow(
        data,
        extent=extent,
        aspect=aspect,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear',
        **kwargs
    )

    # Add wiggle traces if requested
    if wiggle:
        # Normalize data for wiggle display
        trace_width = (distance[-1] - distance[0]) / n_channels
        scale = wiggle_scale * trace_width * 0.8

        for i in range(0, n_channels, wiggle_skip):
            trace = data[:, i]
            trace_norm = trace / (np.abs(data).max() + 1e-10) * scale
            x_wiggle = distance[i] + trace_norm

            # Plot wiggle trace
            ax.plot(
                x_wiggle,
                time,
                color=wiggle_color,
                linewidth=wiggle_linewidth,
                zorder=10
            )

            # Fill positive amplitudes
            if wiggle_fill:
                ax.fill_betweenx(
                    time,
                    distance[i],
                    x_wiggle,
                    where=(trace_norm > 0),
                    color=wiggle_fill_color,
                    alpha=0.6,
                    zorder=9
                )

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig, ax


def plot_spectrogram(
    data: np.ndarray,
    fs: float = 1.0,
    channel: Optional[int] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: Union[str, Tuple] = 'hann',
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    scale: str = 'dB',
    detrend: str = 'constant',
    xlabel: str = 'Time (s)',
    ylabel: str = 'Frequency (Hz)',
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 100,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot spectrogram for frequency analysis of DAS data.

    This function creates publication-quality spectrograms using Short-Time
    Fourier Transform (STFT) to analyze the frequency content of DAS signals
    over time.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array of DAS data. If 2D, must specify channel or it will use mean.
    fs : float, default=1.0
        Sampling frequency in Hz.
    channel : int, optional
        Channel index to plot if data is 2D. If None, uses mean across all channels.
    nperseg : int, optional
        Length of each segment for STFT. If None, uses 256.
    noverlap : int, optional
        Number of points to overlap between segments. If None, uses nperseg // 2.
    window : str or tuple, default='hann'
        Window function to use (e.g., 'hann', 'hamming', 'blackman').
    fmin : float, optional
        Minimum frequency to display. If None, starts from 0 Hz.
    fmax : float, optional
        Maximum frequency to display. If None, uses Nyquist frequency.
    vmin : float, optional
        Minimum value for color scale. If None, auto-scales.
    vmax : float, optional
        Maximum value for color scale. If None, auto-scales.
    cmap : str, default='viridis'
        Colormap name.
    scale : str, default='dB'
        Scale for spectrogram ('dB' for decibels, 'linear' for linear scale).
    detrend : str, default='constant'
        Detrend type ('constant', 'linear', or False).
    xlabel : str, default='Time (s)'
        Label for x-axis.
    ylabel : str, default='Frequency (Hz)'
        Label for y-axis.
    title : str, optional
        Plot title.
    colorbar : bool, default=True
        Whether to add a colorbar.
    colorbar_label : str, optional
        Label for the colorbar. If None, uses 'Power (dB)' or 'Power'.
    figsize : tuple, default=(12, 6)
        Figure size in inches (width, height).
    dpi : int, default=100
        Dots per inch for the figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to pcolormesh().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    f : np.ndarray
        Frequency array.
    t : np.ndarray
        Time array.
    Sxx : np.ndarray
        Spectrogram values.

    Examples
    --------
    >>> import numpy as np
    >>> from das_toolkit import plot_spectrogram
    >>>
    >>> # Generate synthetic signal with multiple frequencies
    >>> fs = 1000  # Hz
    >>> t = np.linspace(0, 10, fs * 10)
    >>> signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    >>>
    >>> # Plot spectrogram
    >>> fig, ax, f, t, Sxx = plot_spectrogram(signal, fs=fs, fmax=200)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Handle multi-channel data
    if data.ndim == 2:
        if channel is not None:
            signal_data = data[:, channel] if data.shape[0] > data.shape[1] else data[channel, :]
        else:
            # Use mean across channels
            signal_data = np.mean(data, axis=1 if data.shape[0] > data.shape[1] else 0)
    else:
        signal_data = data

    # Set default segment length
    if nperseg is None:
        nperseg = min(256, len(signal_data) // 8)

    if noverlap is None:
        noverlap = nperseg // 2

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        signal_data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling='density'
    )

    # Apply frequency limits
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = f[-1]

    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Convert to dB if requested
    if scale == 'dB':
        Sxx_plot = 10 * np.log10(Sxx + 1e-10)
        if colorbar_label is None:
            colorbar_label = 'Power (dB)'
    else:
        Sxx_plot = Sxx
        if colorbar_label is None:
            colorbar_label = 'Power'

    # Auto-scale if limits not provided
    if vmin is None:
        vmin = np.percentile(Sxx_plot, 5)
    if vmax is None:
        vmax = np.percentile(Sxx_plot, 95)

    # Plot spectrogram
    im = ax.pcolormesh(
        t,
        f,
        Sxx_plot,
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs
    )

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig, ax, f, t, Sxx


def plot_velocity_profile(
    depth: np.ndarray,
    velocity: np.ndarray,
    velocity_error: Optional[np.ndarray] = None,
    velocity_profiles: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    markers: Optional[Dict[str, float]] = None,
    marker_labels: Optional[Dict[str, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlabel: str = 'Velocity (m/s)',
    ylabel: str = 'Depth (m)',
    title: Optional[str] = None,
    legend: bool = True,
    legend_loc: str = 'best',
    grid: bool = True,
    invert_yaxis: bool = True,
    error_alpha: float = 0.3,
    linewidth: float = 2.0,
    markersize: float = 6.0,
    figsize: Tuple[float, float] = (8, 10),
    dpi: int = 100,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot velocity profile for VSP (Vertical Seismic Profiling) results.

    This function creates publication-quality velocity-depth profiles commonly
    used in VSP analysis. It supports uncertainty bands, multiple profile
    comparison, and geological layer markers.

    Parameters
    ----------
    depth : np.ndarray
        1D array of depth values.
    velocity : np.ndarray
        1D array of velocity values corresponding to depths.
    velocity_error : np.ndarray, optional
        1D array of velocity uncertainties for error bands.
    velocity_profiles : dict, optional
        Additional velocity profiles to compare. Keys are profile names,
        values are tuples of (depth, velocity) arrays.
    markers : dict, optional
        Depth markers for geological layers. Keys are marker names,
        values are depth values.
    marker_labels : dict, optional
        Custom labels for markers. Keys are marker names, values are label strings.
    vmin : float, optional
        Minimum velocity for x-axis. If None, auto-scales.
    vmax : float, optional
        Maximum velocity for x-axis. If None, auto-scales.
    xlabel : str, default='Velocity (m/s)'
        Label for x-axis.
    ylabel : str, default='Depth (m)'
        Label for y-axis.
    title : str, optional
        Plot title.
    legend : bool, default=True
        Whether to show legend.
    legend_loc : str, default='best'
        Legend location.
    grid : bool, default=True
        Whether to show grid lines.
    invert_yaxis : bool, default=True
        Whether to invert y-axis (depth increases downward).
    error_alpha : float, default=0.3
        Transparency for error bands.
    linewidth : float, default=2.0
        Line width for velocity profiles.
    markersize : float, default=6.0
        Size of data point markers.
    figsize : tuple, default=(8, 10)
        Figure size in inches (width, height).
    dpi : int, default=100
        Dots per inch for the figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to plot().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import numpy as np
    >>> from das_toolkit import plot_velocity_profile
    >>>
    >>> # Generate synthetic VSP data
    >>> depth = np.linspace(0, 1000, 50)
    >>> velocity = 1500 + 0.6 * depth + 50 * np.sin(depth / 100)
    >>> velocity_error = 50 * np.ones_like(velocity)
    >>>
    >>> # Plot single profile with error bars
    >>> fig, ax = plot_velocity_profile(depth, velocity, velocity_error)
    >>>
    >>> # Plot with multiple profiles and markers
    >>> other_profiles = {
    ...     'Model 1': (depth, 1500 + 0.5 * depth),
    ...     'Model 2': (depth, 1600 + 0.7 * depth)
    ... }
    >>> markers = {'Layer 1': 300, 'Layer 2': 600}
    >>> fig, ax = plot_velocity_profile(
    ...     depth, velocity, velocity_error,
    ...     velocity_profiles=other_profiles,
    ...     markers=markers
    ... )
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Plot main velocity profile
    line, = ax.plot(
        velocity,
        depth,
        'o-',
        linewidth=linewidth,
        markersize=markersize,
        label='Measured',
        **kwargs
    )
    main_color = line.get_color()

    # Add error bands if provided
    if velocity_error is not None:
        ax.fill_betweenx(
            depth,
            velocity - velocity_error,
            velocity + velocity_error,
            alpha=error_alpha,
            color=main_color,
            label='Uncertainty'
        )

    # Plot additional velocity profiles if provided
    if velocity_profiles:
        colors = plt.cm.tab10(np.linspace(0, 1, len(velocity_profiles)))
        for (name, (prof_depth, prof_velocity)), color in zip(
            velocity_profiles.items(), colors
        ):
            ax.plot(
                prof_velocity,
                prof_depth,
                '--',
                linewidth=linewidth - 0.5,
                label=name,
                alpha=0.8,
                color=color
            )

    # Add depth markers if provided
    if markers:
        for marker_name, marker_depth in markers.items():
            label = marker_labels.get(marker_name, marker_name) if marker_labels else marker_name
            ax.axhline(
                marker_depth,
                color='gray',
                linestyle=':',
                linewidth=1.5,
                alpha=0.7,
                zorder=1
            )
            # Add text label on the right side
            xlim = ax.get_xlim()
            ax.text(
                xlim[1],
                marker_depth,
                f'  {label}',
                verticalalignment='center',
                fontsize=9,
                alpha=0.8
            )

    # Set velocity limits if provided
    if vmin is not None or vmax is not None:
        current_xlim = ax.get_xlim()
        new_vmin = vmin if vmin is not None else current_xlim[0]
        new_vmax = vmax if vmax is not None else current_xlim[1]
        ax.set_xlim(new_vmin, new_vmax)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    if invert_yaxis:
        ax.invert_yaxis()

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if legend:
        ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9)

    ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig, ax
