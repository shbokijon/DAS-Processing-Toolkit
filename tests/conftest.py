"""
Pytest fixtures for DAS Processing Toolkit tests.
"""

import pytest
import numpy as np


@pytest.fixture
def simple_signal():
    """
    Generate a simple 1D sinusoidal signal.

    Returns
    -------
    ndarray
        1D array with sine wave (100 samples)
    """
    t = np.linspace(0, 1, 100)
    return np.sin(2 * np.pi * 5 * t)


@pytest.fixture
def noisy_signal():
    """
    Generate a noisy 1D signal (sine wave + Gaussian noise).

    Returns
    -------
    ndarray
        1D array with noisy sine wave (100 samples)
    """
    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    clean_signal = np.sin(2 * np.pi * 5 * t)
    noise = np.random.normal(0, 0.5, 100)
    return clean_signal + noise


@pytest.fixture
def multi_frequency_signal():
    """
    Generate a signal with multiple frequency components.

    Returns
    -------
    ndarray
        1D array with multiple frequency components (1000 samples)
    """
    t = np.linspace(0, 1, 1000)
    # 5 Hz + 50 Hz + 100 Hz components
    signal = (np.sin(2 * np.pi * 5 * t) +
              0.5 * np.sin(2 * np.pi * 50 * t) +
              0.3 * np.sin(2 * np.pi * 100 * t))
    return signal


@pytest.fixture
def sampling_rate():
    """
    Standard sampling rate for test signals.

    Returns
    -------
    float
        Sampling rate in Hz
    """
    return 1000.0


@pytest.fixture
def noisy_2d_data():
    """
    Generate noisy 2D data for denoising tests.

    Returns
    -------
    ndarray
        2D array (50x50) with structured signal and noise
    """
    np.random.seed(42)
    # Create a simple 2D pattern
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    clean_data = np.sin(np.sqrt(X**2 + Y**2))

    # Add noise
    noise = np.random.normal(0, 0.3, (50, 50))
    return clean_data + noise


@pytest.fixture
def das_2d_data():
    """
    Generate synthetic DAS data (time x space).

    Returns
    -------
    ndarray
        2D array (100x50) representing DAS data
    """
    np.random.seed(42)
    nt, nx = 100, 50
    t = np.linspace(0, 1, nt)
    x = np.linspace(0, 100, nx)

    # Create a synthetic wavefield with a linear moveout
    data = np.zeros((nt, nx))
    for i, xi in enumerate(x):
        delay = int(xi / 100 * 20)  # Linear moveout
        if delay < nt:
            data[delay:, i] = np.sin(2 * np.pi * 10 * t[:nt-delay])

    # Add noise
    noise = np.random.normal(0, 0.1, (nt, nx))
    return data + noise


@pytest.fixture
def spike_signal():
    """
    Generate a signal with sparse spikes for median filter testing.

    Returns
    -------
    ndarray
        1D array with spikes
    """
    np.random.seed(42)
    signal = np.zeros(100)
    # Add some spikes
    signal[20] = 10
    signal[45] = -8
    signal[70] = 12
    # Add baseline signal
    signal += np.sin(np.linspace(0, 4*np.pi, 100))
    return signal


@pytest.fixture
def empty_array():
    """
    Empty numpy array for edge case testing.

    Returns
    -------
    ndarray
        Empty array
    """
    return np.array([])


@pytest.fixture
def small_array():
    """
    Small array for edge case testing.

    Returns
    -------
    ndarray
        Array with 3 elements
    """
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def das_params():
    """
    Standard DAS data parameters.

    Returns
    -------
    dict
        Dictionary containing dx, dt, and velocity parameters
    """
    return {
        'dx': 2.0,      # 2 meters spatial sampling
        'dt': 0.01,     # 10 ms temporal sampling
        'vmin': 1000.0, # 1000 m/s minimum velocity
        'vmax': 5000.0  # 5000 m/s maximum velocity
    }
