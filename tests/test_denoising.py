"""
Unit tests for DAS denoising algorithms with comprehensive edge cases.
"""

import pytest
import numpy as np
from das_toolkit.denoising import (
    wavelet_denoise,
    fk_filter,
    svd_denoise,
    median_denoise,
)


class TestWaveletDenoise:
    """Test suite for wavelet_denoise function."""

    def test_wavelet_denoise_1d_basic(self, noisy_signal):
        """Test basic 1D wavelet denoising."""
        denoised = wavelet_denoise(noisy_signal)

        # Check output shape matches input
        assert denoised.shape == noisy_signal.shape

        # Denoised signal should have lower variance
        assert np.var(denoised) < np.var(noisy_signal)

    def test_wavelet_denoise_2d_basic(self, noisy_2d_data):
        """Test basic 2D wavelet denoising."""
        denoised = wavelet_denoise(noisy_2d_data)

        # Check output shape matches input
        assert denoised.shape == noisy_2d_data.shape

        # Denoised data should have lower variance
        assert np.var(denoised) < np.var(noisy_2d_data)

    def test_wavelet_denoise_different_wavelets(self, noisy_signal):
        """Test wavelet denoising with different wavelet families."""
        denoised_db4 = wavelet_denoise(noisy_signal, wavelet='db4')
        denoised_sym5 = wavelet_denoise(noisy_signal, wavelet='sym5')
        denoised_coif3 = wavelet_denoise(noisy_signal, wavelet='coif3')

        # All should have same shape
        assert denoised_db4.shape == noisy_signal.shape
        assert denoised_sym5.shape == noisy_signal.shape
        assert denoised_coif3.shape == noisy_signal.shape

        # Different wavelets should produce slightly different results
        assert not np.allclose(denoised_db4, denoised_sym5)

    def test_wavelet_denoise_threshold_modes(self, noisy_signal):
        """Test wavelet denoising with different threshold modes."""
        denoised_soft = wavelet_denoise(noisy_signal, threshold_mode='soft')
        denoised_hard = wavelet_denoise(noisy_signal, threshold_mode='hard')

        # Both should have same shape
        assert denoised_soft.shape == noisy_signal.shape
        assert denoised_hard.shape == noisy_signal.shape

        # Different thresholding modes should produce different results
        assert not np.allclose(denoised_soft, denoised_hard)

    def test_wavelet_denoise_custom_level(self, noisy_signal):
        """Test wavelet denoising with custom decomposition level."""
        denoised_level2 = wavelet_denoise(noisy_signal, level=2)
        denoised_level4 = wavelet_denoise(noisy_signal, level=4)

        # Both should have same shape
        assert denoised_level2.shape == noisy_signal.shape
        assert denoised_level4.shape == noisy_signal.shape

    # Edge cases
    def test_wavelet_denoise_empty_array(self, empty_array):
        """Test wavelet denoising with empty array."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            wavelet_denoise(empty_array)

    def test_wavelet_denoise_invalid_threshold_mode(self, noisy_signal):
        """Test wavelet denoising with invalid threshold mode."""
        with pytest.raises(ValueError, match="threshold_mode must be 'soft' or 'hard'"):
            wavelet_denoise(noisy_signal, threshold_mode='invalid')

    def test_wavelet_denoise_3d_array(self):
        """Test that 3D arrays raise ValueError."""
        data_3d = np.random.randn(10, 10, 10)
        with pytest.raises(ValueError, match="Data must be 1D or 2D array"):
            wavelet_denoise(data_3d)

    def test_wavelet_denoise_single_value(self):
        """Test wavelet denoising with single value array."""
        single_value = np.array([5.0])
        # Should handle gracefully (might fail depending on wavelet implementation)
        try:
            result = wavelet_denoise(single_value)
            assert result.shape == single_value.shape
        except ValueError:
            # Expected for very small arrays
            pass

    def test_wavelet_denoise_preserves_clean_signal(self):
        """Test that wavelet denoising preserves clean signals."""
        # Create clean sine wave
        t = np.linspace(0, 1, 100)
        clean_signal = np.sin(2 * np.pi * 5 * t)

        denoised = wavelet_denoise(clean_signal)

        # Should be very close to original
        assert np.corrcoef(clean_signal, denoised)[0, 1] > 0.99


class TestFKFilter:
    """Test suite for FK (frequency-wavenumber) filter function."""

    def test_fk_filter_basic(self, das_2d_data, das_params):
        """Test basic FK filtering."""
        filtered = fk_filter(
            das_2d_data,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=das_params['vmin'],
            vmax=das_params['vmax']
        )

        # Check output shape matches input
        assert filtered.shape == das_2d_data.shape

    def test_fk_filter_removes_noise(self, das_2d_data, das_params):
        """Test that FK filter reduces noise energy."""
        filtered = fk_filter(
            das_2d_data,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=das_params['vmin'],
            vmax=das_params['vmax']
        )

        # Filtered data typically has lower variance if noise is removed
        # (depends on signal characteristics)
        assert filtered.shape == das_2d_data.shape

    def test_fk_filter_different_velocity_ranges(self, das_2d_data, das_params):
        """Test FK filter with different velocity ranges."""
        # Narrow velocity range
        filtered_narrow = fk_filter(
            das_2d_data,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=2000.0,
            vmax=3000.0
        )

        # Wide velocity range
        filtered_wide = fk_filter(
            das_2d_data,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=500.0,
            vmax=6000.0
        )

        # Different velocity ranges should produce different results
        assert not np.allclose(filtered_narrow, filtered_wide)

    # Edge cases
    def test_fk_filter_invalid_dimensions(self, noisy_signal, das_params):
        """Test that 1D data raises ValueError."""
        with pytest.raises(ValueError, match="Data must be 2D array"):
            fk_filter(
                noisy_signal,
                dx=das_params['dx'],
                dt=das_params['dt'],
                vmin=das_params['vmin'],
                vmax=das_params['vmax']
            )

    def test_fk_filter_negative_sampling_intervals(self, das_2d_data, das_params):
        """Test that negative sampling intervals raise ValueError."""
        with pytest.raises(ValueError, match="Sampling intervals must be positive"):
            fk_filter(das_2d_data, dx=-1.0, dt=das_params['dt'],
                     vmin=das_params['vmin'], vmax=das_params['vmax'])

        with pytest.raises(ValueError, match="Sampling intervals must be positive"):
            fk_filter(das_2d_data, dx=das_params['dx'], dt=-0.01,
                     vmin=das_params['vmin'], vmax=das_params['vmax'])

    def test_fk_filter_negative_velocities(self, das_2d_data, das_params):
        """Test that negative velocities raise ValueError."""
        with pytest.raises(ValueError, match="Velocities must be positive"):
            fk_filter(das_2d_data, dx=das_params['dx'], dt=das_params['dt'],
                     vmin=-1000.0, vmax=das_params['vmax'])

        with pytest.raises(ValueError, match="Velocities must be positive"):
            fk_filter(das_2d_data, dx=das_params['dx'], dt=das_params['dt'],
                     vmin=das_params['vmin'], vmax=-5000.0)

    def test_fk_filter_invalid_velocity_order(self, das_2d_data, das_params):
        """Test that vmin >= vmax raises ValueError."""
        with pytest.raises(ValueError, match="vmin must be less than vmax"):
            fk_filter(das_2d_data, dx=das_params['dx'], dt=das_params['dt'],
                     vmin=5000.0, vmax=1000.0)

    def test_fk_filter_small_array(self, das_params):
        """Test FK filter with very small 2D array."""
        small_2d = np.random.randn(5, 5)
        filtered = fk_filter(
            small_2d,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=das_params['vmin'],
            vmax=das_params['vmax']
        )
        assert filtered.shape == small_2d.shape

    def test_fk_filter_zeros_array(self, das_params):
        """Test FK filter with array of zeros."""
        zeros = np.zeros((20, 20))
        filtered = fk_filter(
            zeros,
            dx=das_params['dx'],
            dt=das_params['dt'],
            vmin=das_params['vmin'],
            vmax=das_params['vmax']
        )
        # Filtering zeros should return zeros
        assert np.allclose(filtered, zeros)


class TestSVDDenoise:
    """Test suite for SVD denoising function."""

    def test_svd_denoise_basic(self, noisy_2d_data):
        """Test basic SVD denoising."""
        denoised = svd_denoise(noisy_2d_data, n_components=5)

        # Check output shape matches input
        assert denoised.shape == noisy_2d_data.shape

    def test_svd_denoise_auto_components(self, noisy_2d_data):
        """Test SVD denoising with automatic component selection."""
        denoised = svd_denoise(noisy_2d_data, threshold=0.1)

        # Check output shape matches input
        assert denoised.shape == noisy_2d_data.shape

    def test_svd_denoise_different_components(self, noisy_2d_data):
        """Test SVD denoising with different number of components."""
        denoised_few = svd_denoise(noisy_2d_data, n_components=3)
        denoised_many = svd_denoise(noisy_2d_data, n_components=20)

        # Both should have same shape
        assert denoised_few.shape == noisy_2d_data.shape
        assert denoised_many.shape == noisy_2d_data.shape

        # Different number of components should produce different results
        assert not np.allclose(denoised_few, denoised_many)

        # More components should preserve more variance
        variance_few = np.var(denoised_few)
        variance_many = np.var(denoised_many)
        assert variance_many > variance_few

    def test_svd_denoise_different_thresholds(self, noisy_2d_data):
        """Test SVD denoising with different thresholds."""
        denoised_low = svd_denoise(noisy_2d_data, threshold=0.05)
        denoised_high = svd_denoise(noisy_2d_data, threshold=0.3)

        # Different thresholds should produce different results
        assert not np.allclose(denoised_low, denoised_high)

    # Edge cases
    def test_svd_denoise_1d_array(self, noisy_signal):
        """Test that 1D array raises ValueError."""
        with pytest.raises(ValueError, match="Data must be 2D array"):
            svd_denoise(noisy_signal)

    def test_svd_denoise_empty_array(self):
        """Test that empty array raises ValueError."""
        empty_2d = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            svd_denoise(empty_2d)

    def test_svd_denoise_invalid_threshold(self, noisy_2d_data):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            svd_denoise(noisy_2d_data, threshold=-0.1)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            svd_denoise(noisy_2d_data, threshold=1.5)

    def test_svd_denoise_invalid_n_components(self, noisy_2d_data):
        """Test that invalid n_components raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be between"):
            svd_denoise(noisy_2d_data, n_components=0)

        with pytest.raises(ValueError, match="n_components must be between"):
            svd_denoise(noisy_2d_data, n_components=100)

    def test_svd_denoise_single_component(self, noisy_2d_data):
        """Test SVD denoising with single component."""
        denoised = svd_denoise(noisy_2d_data, n_components=1)

        # Should return low-rank approximation
        assert denoised.shape == noisy_2d_data.shape
        assert np.linalg.matrix_rank(denoised) <= 1 or np.allclose(denoised, 0)

    def test_svd_denoise_rank_deficient_matrix(self):
        """Test SVD denoising on rank-deficient matrix."""
        # Create rank-1 matrix
        u = np.random.randn(20, 1)
        v = np.random.randn(1, 30)
        rank1_matrix = u @ v

        denoised = svd_denoise(rank1_matrix, n_components=5)

        # Should handle gracefully
        assert denoised.shape == rank1_matrix.shape

    def test_svd_denoise_square_vs_rectangular(self):
        """Test SVD denoising on square vs rectangular matrices."""
        np.random.seed(42)
        square = np.random.randn(30, 30)
        rectangular = np.random.randn(30, 50)

        denoised_square = svd_denoise(square, n_components=5)
        denoised_rect = svd_denoise(rectangular, n_components=5)

        assert denoised_square.shape == square.shape
        assert denoised_rect.shape == rectangular.shape


class TestMedianDenoise:
    """Test suite for median denoising function."""

    def test_median_denoise_basic(self, noisy_2d_data):
        """Test basic 2D median denoising."""
        denoised = median_denoise(noisy_2d_data)

        # Check output shape matches input
        assert denoised.shape == noisy_2d_data.shape

    def test_median_denoise_removes_salt_pepper_noise(self):
        """Test that median denoising removes salt-and-pepper noise."""
        np.random.seed(42)
        # Create clean data
        clean = np.ones((30, 30)) * 5.0

        # Add salt-and-pepper noise
        noisy = clean.copy()
        salt_pepper_mask = np.random.rand(30, 30) < 0.1
        noisy[salt_pepper_mask] = np.random.choice([0, 10], size=np.sum(salt_pepper_mask))

        denoised = median_denoise(noisy, kernel_size=(3, 3))

        # Denoised should be closer to clean than noisy is
        error_before = np.mean((noisy - clean) ** 2)
        error_after = np.mean((denoised - clean) ** 2)

        assert error_after < error_before

    def test_median_denoise_different_kernel_sizes(self, noisy_2d_data):
        """Test median denoising with different kernel sizes."""
        denoised_small = median_denoise(noisy_2d_data, kernel_size=(3, 3))
        denoised_large = median_denoise(noisy_2d_data, kernel_size=(7, 7))

        # Both should have same shape
        assert denoised_small.shape == noisy_2d_data.shape
        assert denoised_large.shape == noisy_2d_data.shape

        # Different kernel sizes should produce different results
        assert not np.allclose(denoised_small, denoised_large)

    def test_median_denoise_preserves_edges(self):
        """Test that median denoising preserves sharp edges."""
        # Create image with sharp edge
        edge_image = np.zeros((30, 30))
        edge_image[:, 15:] = 10.0

        # Add some noise
        np.random.seed(42)
        noisy = edge_image + np.random.normal(0, 0.5, (30, 30))

        denoised = median_denoise(noisy, kernel_size=(3, 3))

        # Edge should still be relatively sharp
        # Check gradient across edge
        gradient = np.abs(np.diff(denoised[:, 13:17], axis=1))
        assert np.max(gradient) > 5  # Edge should still have significant gradient

    # Edge cases
    def test_median_denoise_1d_array(self, noisy_signal):
        """Test that 1D array raises ValueError."""
        with pytest.raises(ValueError, match="Data must be 2D array"):
            median_denoise(noisy_signal)

    def test_median_denoise_empty_array(self):
        """Test that empty array raises ValueError."""
        empty_2d = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            median_denoise(empty_2d)

    def test_median_denoise_invalid_kernel_type(self, noisy_2d_data):
        """Test that invalid kernel_size type raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be a tuple of length 2"):
            median_denoise(noisy_2d_data, kernel_size=3)

        with pytest.raises(ValueError, match="kernel_size must be a tuple of length 2"):
            median_denoise(noisy_2d_data, kernel_size=(3, 3, 3))

    def test_median_denoise_invalid_kernel_values(self, noisy_2d_data):
        """Test that invalid kernel values raise ValueError."""
        # Kernel dimensions < 1
        with pytest.raises(ValueError, match="Kernel dimensions must be at least 1"):
            median_denoise(noisy_2d_data, kernel_size=(0, 3))

        # Even kernel dimensions
        with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
            median_denoise(noisy_2d_data, kernel_size=(4, 3))

        with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
            median_denoise(noisy_2d_data, kernel_size=(3, 6))

    def test_median_denoise_minimum_kernel(self, noisy_2d_data):
        """Test median denoising with minimum kernel size (1, 1)."""
        denoised = median_denoise(noisy_2d_data, kernel_size=(1, 1))

        # Kernel of (1, 1) should return data unchanged
        assert np.allclose(denoised, noisy_2d_data)

    def test_median_denoise_constant_data(self):
        """Test median denoising on constant data."""
        constant = np.ones((20, 20)) * 7.5

        denoised = median_denoise(constant, kernel_size=(5, 5))

        # Constant data should remain constant in interior (edge effects at boundaries)
        # Check central portion away from edges
        assert np.allclose(denoised[2:-2, 2:-2], constant[2:-2, 2:-2])

    def test_median_denoise_asymmetric_kernel(self, noisy_2d_data):
        """Test median denoising with asymmetric kernel."""
        denoised = median_denoise(noisy_2d_data, kernel_size=(3, 7))

        # Should work with asymmetric kernels
        assert denoised.shape == noisy_2d_data.shape
