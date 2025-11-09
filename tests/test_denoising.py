"""
Comprehensive tests for wavelet denoising module.
"""

import numpy as np
import pytest
import pywt
from das_toolkit.denoising import (
    wavelet_denoise,
    multilevel_wavelet_denoise,
    compare_denoising_methods,
    soft_threshold,
    hard_threshold,
    estimate_noise_std,
    calculate_threshold,
    calculate_snr,
    calculate_rmse,
    denoise_2d_signal,
)


class TestThresholding:
    """Test thresholding functions."""

    def test_soft_threshold_basic(self):
        """Test soft thresholding with simple values."""
        coeffs = np.array([-3, -1, 0, 1, 3])
        threshold = 1.5
        result = soft_threshold(coeffs, threshold)

        expected = np.array([-1.5, 0, 0, 0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hard_threshold_basic(self):
        """Test hard thresholding with simple values."""
        coeffs = np.array([-3, -1, 0, 1, 3])
        threshold = 1.5
        result = hard_threshold(coeffs, threshold)

        expected = np.array([-3, 0, 0, 0, 3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_soft_threshold_preserves_shape(self):
        """Test that soft thresholding preserves array shape."""
        coeffs = np.random.randn(100)
        result = soft_threshold(coeffs, 0.5)
        assert result.shape == coeffs.shape

    def test_hard_threshold_preserves_shape(self):
        """Test that hard thresholding preserves array shape."""
        coeffs = np.random.randn(100)
        result = hard_threshold(coeffs, 0.5)
        assert result.shape == coeffs.shape


class TestNoiseEstimation:
    """Test noise estimation functions."""

    def test_estimate_noise_std_mad(self):
        """Test MAD-based noise estimation."""
        # Create coefficients with known noise level
        np.random.seed(42)
        noise_level = 0.5
        coeffs = np.random.normal(0, noise_level, 1000)

        estimated_std = estimate_noise_std(coeffs, method='mad')

        # Should be close to actual noise level
        assert abs(estimated_std - noise_level) < 0.1

    def test_estimate_noise_std_std(self):
        """Test standard deviation-based noise estimation."""
        np.random.seed(42)
        noise_level = 0.5
        coeffs = np.random.normal(0, noise_level, 1000)

        estimated_std = estimate_noise_std(coeffs, method='std')

        # Should be close to actual noise level
        assert abs(estimated_std - noise_level) < 0.1

    def test_estimate_noise_std_invalid_method(self):
        """Test that invalid method raises error."""
        coeffs = np.random.randn(100)
        with pytest.raises(ValueError):
            estimate_noise_std(coeffs, method='invalid')


class TestThresholdCalculation:
    """Test threshold calculation methods."""

    def test_universal_threshold(self):
        """Test universal threshold calculation."""
        np.random.seed(42)
        N = 1000
        noise_std = 1.0
        coeffs = np.random.normal(0, noise_std, N)

        threshold = calculate_threshold(coeffs, method='universal', noise_std=noise_std)

        # Universal threshold formula: sigma * sqrt(2 * log(N))
        expected = noise_std * np.sqrt(2 * np.log(N))
        assert abs(threshold - expected) < 0.01

    def test_minimax_threshold(self):
        """Test minimax threshold calculation."""
        np.random.seed(42)
        coeffs = np.random.randn(1000)
        threshold = calculate_threshold(coeffs, method='minimax')

        # Should return a positive value
        assert threshold > 0

    def test_sure_threshold(self):
        """Test SURE threshold calculation."""
        np.random.seed(42)
        coeffs = np.random.randn(1000)
        threshold = calculate_threshold(coeffs, method='sure')

        # Should return a positive value
        assert threshold >= 0

    def test_threshold_invalid_method(self):
        """Test that invalid method raises error."""
        coeffs = np.random.randn(100)
        with pytest.raises(ValueError):
            calculate_threshold(coeffs, method='invalid')


class TestWaveletDenoise:
    """Test main wavelet denoising function."""

    def setup_method(self):
        """Set up test signals."""
        np.random.seed(42)
        self.t = np.linspace(0, 1, 1000)

        # Create clean signal (sum of sinusoids)
        self.clean_signal = (
            np.sin(2 * np.pi * 5 * self.t) +
            0.5 * np.sin(2 * np.pi * 10 * self.t)
        )

        # Add noise
        self.noise_level = 0.3
        self.noisy_signal = self.clean_signal + np.random.normal(
            0, self.noise_level, len(self.clean_signal)
        )

    def test_denoise_improves_snr(self):
        """Test that denoising improves SNR."""
        denoised = wavelet_denoise(self.noisy_signal, wavelet='db4')

        snr_noisy = calculate_snr(self.clean_signal, self.noisy_signal)
        snr_denoised = calculate_snr(self.clean_signal, denoised)

        # Denoised signal should have better SNR
        assert snr_denoised > snr_noisy

    def test_denoise_reduces_rmse(self):
        """Test that denoising reduces RMSE."""
        denoised = wavelet_denoise(self.noisy_signal, wavelet='db4')

        rmse_noisy = calculate_rmse(self.clean_signal, self.noisy_signal)
        rmse_denoised = calculate_rmse(self.clean_signal, denoised)

        # Denoised signal should have lower RMSE
        assert rmse_denoised < rmse_noisy

    def test_denoise_preserves_length(self):
        """Test that denoising preserves signal length."""
        denoised = wavelet_denoise(self.noisy_signal, wavelet='db4')
        assert len(denoised) == len(self.noisy_signal)

    def test_different_wavelets(self):
        """Test denoising with different wavelet families."""
        wavelets = ['db4', 'sym4', 'coif3', 'haar']

        for wavelet in wavelets:
            denoised = wavelet_denoise(self.noisy_signal, wavelet=wavelet)
            assert len(denoised) == len(self.noisy_signal)

            # Should improve SNR
            snr_noisy = calculate_snr(self.clean_signal, self.noisy_signal)
            snr_denoised = calculate_snr(self.clean_signal, denoised)
            assert snr_denoised > snr_noisy

    def test_soft_vs_hard_thresholding(self):
        """Test both soft and hard thresholding modes."""
        soft_denoised = wavelet_denoise(
            self.noisy_signal, threshold_mode='soft'
        )
        hard_denoised = wavelet_denoise(
            self.noisy_signal, threshold_mode='hard'
        )

        # Both should improve SNR
        snr_noisy = calculate_snr(self.clean_signal, self.noisy_signal)
        snr_soft = calculate_snr(self.clean_signal, soft_denoised)
        snr_hard = calculate_snr(self.clean_signal, hard_denoised)

        assert snr_soft > snr_noisy
        assert snr_hard > snr_noisy

    def test_different_threshold_methods(self):
        """Test different threshold calculation methods."""
        methods = ['universal', 'sure', 'minimax']

        for method in methods:
            denoised = wavelet_denoise(
                self.noisy_signal, threshold_method=method
            )
            assert len(denoised) == len(self.noisy_signal)

    def test_custom_level(self):
        """Test denoising with custom decomposition level."""
        for level in [1, 3, 5]:
            denoised = wavelet_denoise(self.noisy_signal, level=level)
            assert len(denoised) == len(self.noisy_signal)

    def test_invalid_threshold_mode(self):
        """Test that invalid threshold mode raises error."""
        with pytest.raises(ValueError):
            wavelet_denoise(self.noisy_signal, threshold_mode='invalid')


class TestMultilevelDenoise:
    """Test multi-level wavelet denoising."""

    def setup_method(self):
        """Set up test signal."""
        np.random.seed(42)
        t = np.linspace(0, 1, 1000)
        self.signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.2, len(t))

    def test_multilevel_single_result(self):
        """Test multilevel denoising returning single result."""
        denoised = multilevel_wavelet_denoise(
            self.signal, return_all_levels=False
        )
        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(self.signal)

    def test_multilevel_all_levels(self):
        """Test multilevel denoising returning all levels."""
        results = multilevel_wavelet_denoise(
            self.signal, return_all_levels=True
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # All results should have same length
        for level, denoised in results.items():
            assert len(denoised) == len(self.signal)
            assert isinstance(level, int)

    def test_multilevel_custom_max_level(self):
        """Test multilevel denoising with custom max level."""
        max_level = 3
        results = multilevel_wavelet_denoise(
            self.signal, max_level=max_level, return_all_levels=True
        )

        assert len(results) == max_level
        assert max(results.keys()) == max_level


class TestCompareMethods:
    """Test comparison of denoising methods."""

    def setup_method(self):
        """Set up test signal."""
        np.random.seed(42)
        t = np.linspace(0, 1, 1000)
        self.signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.2, len(t))

    def test_compare_basic(self):
        """Test basic comparison functionality."""
        results = compare_denoising_methods(self.signal)

        assert isinstance(results, dict)
        assert len(results) > 0

        # All results should have same length as input
        for method_name, denoised in results.items():
            assert len(denoised) == len(self.signal)
            assert isinstance(method_name, str)

    def test_compare_custom_wavelets(self):
        """Test comparison with custom wavelet list."""
        wavelets = ['db4', 'haar']
        results = compare_denoising_methods(
            self.signal, wavelets=wavelets
        )

        # Should have results for both wavelets
        assert any('db4' in key for key in results.keys())
        assert any('haar' in key for key in results.keys())

    def test_compare_custom_modes(self):
        """Test comparison with custom threshold modes."""
        threshold_modes = ['soft', 'hard']
        results = compare_denoising_methods(
            self.signal, threshold_modes=threshold_modes
        )

        # Should have results for both modes
        assert any('soft' in key for key in results.keys())
        assert any('hard' in key for key in results.keys())

    def test_compare_custom_methods(self):
        """Test comparison with custom threshold methods."""
        threshold_methods = ['universal', 'sure']
        results = compare_denoising_methods(
            self.signal,
            wavelets=['db4'],
            threshold_modes=['soft'],
            threshold_methods=threshold_methods
        )

        # Should have results for both threshold methods
        assert any('universal' in key for key in results.keys())
        assert any('sure' in key for key in results.keys())


class TestMetrics:
    """Test metric calculation functions."""

    def test_snr_perfect_signal(self):
        """Test SNR with identical signals."""
        signal = np.random.randn(100)
        snr = calculate_snr(signal, signal)

        # Should be infinite for identical signals
        assert np.isinf(snr)

    def test_snr_calculation(self):
        """Test SNR calculation with known noise."""
        np.random.seed(42)
        clean = np.ones(1000)
        noise = np.random.normal(0, 0.1, 1000)
        noisy = clean + noise

        snr = calculate_snr(clean, noisy)

        # SNR should be positive for low noise
        assert snr > 0

    def test_rmse_identical_signals(self):
        """Test RMSE with identical signals."""
        signal = np.random.randn(100)
        rmse = calculate_rmse(signal, signal)

        # Should be zero for identical signals
        assert rmse == 0

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        clean = np.zeros(100)
        noisy = np.ones(100)

        rmse = calculate_rmse(clean, noisy)

        # RMSE should be 1.0
        assert abs(rmse - 1.0) < 1e-10

    def test_rmse_positive(self):
        """Test that RMSE is always positive."""
        np.random.seed(42)
        clean = np.random.randn(100)
        noisy = clean + np.random.randn(100)

        rmse = calculate_rmse(clean, noisy)
        assert rmse >= 0


class Test2DDenoise:
    """Test 2D signal denoising (for DAS data)."""

    def setup_method(self):
        """Set up 2D test signal."""
        np.random.seed(42)
        n_channels = 10
        n_samples = 1000

        # Create 2D signal
        t = np.linspace(0, 1, n_samples)
        self.clean_2d = np.array([
            np.sin(2 * np.pi * (5 + i * 0.5) * t)
            for i in range(n_channels)
        ])

        # Add noise
        self.noisy_2d = self.clean_2d + np.random.normal(0, 0.2, self.clean_2d.shape)

    def test_denoise_2d_shape_preservation(self):
        """Test that 2D denoising preserves shape."""
        denoised = denoise_2d_signal(self.noisy_2d, axis=1)
        assert denoised.shape == self.noisy_2d.shape

    def test_denoise_2d_axis_0(self):
        """Test 2D denoising along axis 0."""
        denoised = denoise_2d_signal(self.noisy_2d, axis=0)
        assert denoised.shape == self.noisy_2d.shape

    def test_denoise_2d_axis_1(self):
        """Test 2D denoising along axis 1 (time axis)."""
        denoised = denoise_2d_signal(self.noisy_2d, axis=1)
        assert denoised.shape == self.noisy_2d.shape

        # Should improve overall RMSE
        rmse_before = calculate_rmse(self.clean_2d.ravel(), self.noisy_2d.ravel())
        rmse_after = calculate_rmse(self.clean_2d.ravel(), denoised.ravel())
        assert rmse_after < rmse_before

    def test_denoise_2d_improves_quality(self):
        """Test that 2D denoising improves signal quality."""
        denoised = denoise_2d_signal(self.noisy_2d, axis=1)

        # Calculate SNR improvement
        snr_before = calculate_snr(self.clean_2d.ravel(), self.noisy_2d.ravel())
        snr_after = calculate_snr(self.clean_2d.ravel(), denoised.ravel())

        assert snr_after > snr_before


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_signal(self):
        """Test denoising with very short signal."""
        short_signal = np.random.randn(10)
        denoised = wavelet_denoise(short_signal, level=1)
        assert len(denoised) == len(short_signal)

    def test_constant_signal(self):
        """Test denoising with constant signal."""
        constant = np.ones(100)
        denoised = wavelet_denoise(constant)

        # Should remain approximately constant
        assert np.allclose(denoised, constant, atol=0.1)

    def test_zero_signal(self):
        """Test denoising with zero signal."""
        zeros = np.zeros(100)
        denoised = wavelet_denoise(zeros)

        # Should remain zero
        assert np.allclose(denoised, zeros, atol=1e-10)

    def test_large_signal(self):
        """Test denoising with large signal."""
        large_signal = np.random.randn(10000)
        denoised = wavelet_denoise(large_signal)
        assert len(denoised) == len(large_signal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
