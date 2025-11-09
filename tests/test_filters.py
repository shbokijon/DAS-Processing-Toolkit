"""
Unit tests for DAS signal filtering functions.
"""

import pytest
import numpy as np
from das_toolkit.filters import (
    bandpass_filter,
    lowpass_filter,
    highpass_filter,
    median_filter,
    moving_average_filter,
)


class TestBandpassFilter:
    """Test suite for bandpass_filter function."""

    def test_bandpass_filter_basic(self, multi_frequency_signal, sampling_rate):
        """Test basic bandpass filtering functionality."""
        # Filter to keep only 40-60 Hz
        filtered = bandpass_filter(
            multi_frequency_signal,
            lowcut=40,
            highcut=60,
            fs=sampling_rate
        )

        # Check output shape matches input
        assert filtered.shape == multi_frequency_signal.shape

        # Check output is not identical to input
        assert not np.allclose(filtered, multi_frequency_signal)

    def test_bandpass_filter_frequency_response(self, sampling_rate):
        """Test that bandpass filter attenuates frequencies outside band."""
        # Create signal with known frequencies
        t = np.linspace(0, 1, int(sampling_rate))
        low_freq = np.sin(2 * np.pi * 5 * t)     # 5 Hz (should be removed)
        mid_freq = np.sin(2 * np.pi * 50 * t)    # 50 Hz (should be kept)
        high_freq = np.sin(2 * np.pi * 200 * t)  # 200 Hz (should be removed)

        signal = low_freq + mid_freq + high_freq

        # Apply bandpass filter to keep 40-60 Hz
        filtered = bandpass_filter(signal, lowcut=40, highcut=60, fs=sampling_rate)

        # Compute FFT to check frequency content
        fft_original = np.abs(np.fft.fft(signal))
        fft_filtered = np.abs(np.fft.fft(filtered))

        # The 50 Hz component should dominate in filtered signal
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        idx_50hz = np.argmin(np.abs(freqs - 50))

        # Power at 50 Hz should be significant in filtered signal
        assert fft_filtered[idx_50hz] > 0.5 * np.max(fft_filtered)

    def test_bandpass_filter_invalid_frequencies(self, simple_signal, sampling_rate):
        """Test that invalid frequency parameters raise ValueError."""
        # Negative lowcut
        with pytest.raises(ValueError, match="Cutoff frequencies must be positive"):
            bandpass_filter(simple_signal, lowcut=-10, highcut=50, fs=sampling_rate)

        # Negative highcut
        with pytest.raises(ValueError, match="Cutoff frequencies must be positive"):
            bandpass_filter(simple_signal, lowcut=10, highcut=-50, fs=sampling_rate)

        # lowcut >= highcut
        with pytest.raises(ValueError, match="Low cutoff must be less than high cutoff"):
            bandpass_filter(simple_signal, lowcut=100, highcut=50, fs=sampling_rate)

        # highcut >= Nyquist
        with pytest.raises(ValueError, match="High cutoff must be less than Nyquist frequency"):
            bandpass_filter(simple_signal, lowcut=10, highcut=600, fs=sampling_rate)

    def test_bandpass_filter_different_orders(self, multi_frequency_signal, sampling_rate):
        """Test bandpass filter with different filter orders."""
        filtered_order3 = bandpass_filter(
            multi_frequency_signal, lowcut=40, highcut=60, fs=sampling_rate, order=3
        )
        filtered_order8 = bandpass_filter(
            multi_frequency_signal, lowcut=40, highcut=60, fs=sampling_rate, order=8
        )

        # Different orders should produce different results
        assert not np.allclose(filtered_order3, filtered_order8)

        # Both should have same shape
        assert filtered_order3.shape == filtered_order8.shape


class TestLowpassFilter:
    """Test suite for lowpass_filter function."""

    def test_lowpass_filter_basic(self, multi_frequency_signal, sampling_rate):
        """Test basic lowpass filtering functionality."""
        filtered = lowpass_filter(multi_frequency_signal, cutoff=20, fs=sampling_rate)

        # Check output shape matches input
        assert filtered.shape == multi_frequency_signal.shape

        # Filtered signal should be smoother (lower variance)
        assert np.var(filtered) < np.var(multi_frequency_signal)

    def test_lowpass_filter_removes_high_frequencies(self, sampling_rate):
        """Test that lowpass filter removes high frequency components."""
        t = np.linspace(0, 1, int(sampling_rate))
        low_freq = np.sin(2 * np.pi * 5 * t)      # 5 Hz (should be kept)
        high_freq = np.sin(2 * np.pi * 200 * t)   # 200 Hz (should be removed)
        signal = low_freq + high_freq

        # Apply lowpass filter with 20 Hz cutoff
        filtered = lowpass_filter(signal, cutoff=20, fs=sampling_rate)

        # Filtered signal should be much closer to low frequency component
        assert np.corrcoef(filtered, low_freq)[0, 1] > 0.9

    def test_lowpass_filter_invalid_parameters(self, simple_signal, sampling_rate):
        """Test that invalid parameters raise ValueError."""
        # Negative cutoff
        with pytest.raises(ValueError, match="Cutoff frequency must be positive"):
            lowpass_filter(simple_signal, cutoff=-10, fs=sampling_rate)

        # Cutoff >= Nyquist
        with pytest.raises(ValueError, match="Cutoff must be less than Nyquist frequency"):
            lowpass_filter(simple_signal, cutoff=600, fs=sampling_rate)

    def test_lowpass_filter_preserves_dc(self, sampling_rate):
        """Test that lowpass filter preserves DC component."""
        signal = np.ones(100) * 5.0  # DC signal
        filtered = lowpass_filter(signal, cutoff=50, fs=sampling_rate)

        # DC component should be preserved
        assert np.allclose(filtered, signal, rtol=0.01)


class TestHighpassFilter:
    """Test suite for highpass_filter function."""

    def test_highpass_filter_basic(self, multi_frequency_signal, sampling_rate):
        """Test basic highpass filtering functionality."""
        filtered = highpass_filter(multi_frequency_signal, cutoff=80, fs=sampling_rate)

        # Check output shape matches input
        assert filtered.shape == multi_frequency_signal.shape

    def test_highpass_filter_removes_low_frequencies(self, sampling_rate):
        """Test that highpass filter removes low frequency components."""
        t = np.linspace(0, 1, int(sampling_rate))
        low_freq = np.sin(2 * np.pi * 5 * t)      # 5 Hz (should be removed)
        high_freq = np.sin(2 * np.pi * 200 * t)   # 200 Hz (should be kept)
        signal = low_freq + high_freq

        # Apply highpass filter with 100 Hz cutoff
        filtered = highpass_filter(signal, cutoff=100, fs=sampling_rate)

        # Filtered signal should be much closer to high frequency component
        assert np.corrcoef(filtered, high_freq)[0, 1] > 0.9

    def test_highpass_filter_removes_dc(self, sampling_rate):
        """Test that highpass filter removes DC component."""
        t = np.linspace(0, 1, int(sampling_rate))
        ac_signal = np.sin(2 * np.pi * 100 * t)
        dc_offset = 10.0
        signal = ac_signal + dc_offset

        # Apply highpass filter
        filtered = highpass_filter(signal, cutoff=50, fs=sampling_rate)

        # DC component should be removed
        assert abs(np.mean(filtered)) < 0.1

    def test_highpass_filter_invalid_parameters(self, simple_signal, sampling_rate):
        """Test that invalid parameters raise ValueError."""
        # Negative cutoff
        with pytest.raises(ValueError, match="Cutoff frequency must be positive"):
            highpass_filter(simple_signal, cutoff=-10, fs=sampling_rate)

        # Cutoff >= Nyquist
        with pytest.raises(ValueError, match="Cutoff must be less than Nyquist frequency"):
            highpass_filter(simple_signal, cutoff=600, fs=sampling_rate)


class TestMedianFilter:
    """Test suite for median_filter function."""

    def test_median_filter_removes_spikes(self, spike_signal):
        """Test that median filter effectively removes outliers."""
        filtered = median_filter(spike_signal, kernel_size=5)

        # Check output shape matches input
        assert filtered.shape == spike_signal.shape

        # Maximum value should be reduced (spikes removed)
        assert np.max(filtered) < np.max(spike_signal)

    def test_median_filter_preserves_edges(self):
        """Test that median filter preserves step edges."""
        # Create step signal
        signal = np.concatenate([np.ones(50) * 0, np.ones(50) * 10])

        filtered = median_filter(signal, kernel_size=5)

        # Step should still be present
        assert np.mean(filtered[:40]) < 2
        assert np.mean(filtered[60:]) > 8

    def test_median_filter_invalid_kernel_size(self, simple_signal):
        """Test that invalid kernel sizes raise ValueError."""
        # Kernel size < 1
        with pytest.raises(ValueError, match="Kernel size must be at least 1"):
            median_filter(simple_signal, kernel_size=0)

        # Even kernel size
        with pytest.raises(ValueError, match="Kernel size must be odd"):
            median_filter(simple_signal, kernel_size=4)

    def test_median_filter_different_kernel_sizes(self, spike_signal):
        """Test median filter with different kernel sizes."""
        filtered_small = median_filter(spike_signal, kernel_size=3)
        filtered_large = median_filter(spike_signal, kernel_size=11)

        # Larger kernel should produce smoother output
        variance_small = np.var(np.diff(filtered_small))
        variance_large = np.var(np.diff(filtered_large))

        assert variance_large < variance_small


class TestMovingAverageFilter:
    """Test suite for moving_average_filter function."""

    def test_moving_average_basic(self, noisy_signal):
        """Test basic moving average functionality."""
        filtered = moving_average_filter(noisy_signal, window_size=5)

        # Check output shape matches input
        assert filtered.shape == noisy_signal.shape

        # Filtered signal should be smoother
        assert np.var(filtered) < np.var(noisy_signal)

    def test_moving_average_smoothing_effect(self):
        """Test that moving average smooths the signal."""
        # Create signal with high-frequency noise
        np.random.seed(42)
        signal = np.random.randn(100)

        filtered = moving_average_filter(signal, window_size=11)

        # Measure smoothness by looking at consecutive differences
        diff_original = np.abs(np.diff(signal))
        diff_filtered = np.abs(np.diff(filtered))

        # Filtered signal should have smaller consecutive differences
        assert np.mean(diff_filtered) < np.mean(diff_original)

    def test_moving_average_constant_signal(self):
        """Test that moving average preserves constant signals."""
        signal = np.ones(100) * 5.0

        filtered = moving_average_filter(signal, window_size=5)

        # Constant signal should remain constant in interior (edge effects at boundaries)
        # Check central portion away from edges
        assert np.allclose(filtered[5:-5], signal[5:-5])

    def test_moving_average_invalid_parameters(self, simple_signal):
        """Test that invalid parameters raise ValueError."""
        # Window size < 1
        with pytest.raises(ValueError, match="Window size must be at least 1"):
            moving_average_filter(simple_signal, window_size=0)

        # Window size > data length
        with pytest.raises(ValueError, match="Window size cannot exceed data length"):
            moving_average_filter(simple_signal, window_size=200)

    def test_moving_average_different_window_sizes(self, noisy_signal):
        """Test moving average with different window sizes."""
        filtered_small = moving_average_filter(noisy_signal, window_size=3)
        filtered_large = moving_average_filter(noisy_signal, window_size=15)

        # Larger window should produce smoother output
        assert np.var(filtered_large) < np.var(filtered_small)

    def test_moving_average_linear_signal(self):
        """Test moving average on linear trend."""
        signal = np.linspace(0, 10, 100)

        filtered = moving_average_filter(signal, window_size=5)

        # Linear trend should be mostly preserved (except edges)
        # Check middle portion
        assert np.allclose(filtered[10:-10], signal[10:-10], rtol=0.01)
