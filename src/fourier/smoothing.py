"""輪郭点のフーリエ平滑化処理を定義する。"""

import numpy as np


def smooth_contour(points: np.ndarray, num_freqs: int) -> np.ndarray:
    """
    輪郭点をフーリエ変換で平滑化する。

    Args:
        points:
            輪郭点列
        num_freqs:
            残す周波数成分数

    Returns:
        平滑化後の輪郭点

    Side Effects:
        なし
    """
    if len(points) <= num_freqs:
        return points

    complex_points = points[:, 0] + 1j * points[:, 1]
    fft_coefficients = np.fft.fft(complex_points)
    filtered_coefficients = np.zeros_like(fft_coefficients)
    half_frequency_count = num_freqs // 2

    filtered_coefficients[:half_frequency_count] = fft_coefficients[:half_frequency_count]
    filtered_coefficients[-half_frequency_count:] = fft_coefficients[-half_frequency_count:]

    smoothed_complex_points = np.fft.ifft(filtered_coefficients)
    smoothed_points = np.column_stack(
        (np.real(smoothed_complex_points), np.imag(smoothed_complex_points))
    )
    return smoothed_points.astype(np.int32)
