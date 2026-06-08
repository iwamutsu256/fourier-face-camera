"""フーリエ平滑化処理の基本動作を確認するテスト。"""

import unittest

import numpy as np

from src.config import settings
from src.fourier.smoothing import smooth_contour


class FourierSmoothingTest(unittest.TestCase):
    """フーリエ平滑化を確認する。"""

    def test_returns_original_points_when_points_are_too_few(self):
        """
        輪郭点が周波数数以下の場合に元の点列を返すことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        points = np.array([[0, 0], [1, 1]], dtype=np.int32)

        result = smooth_contour(points, settings.DEFAULT_NUM_FREQUENCIES)

        np.testing.assert_array_equal(points, result)

    def test_preserves_point_count_after_smoothing(self):
        """
        平滑化後も輪郭点数が変わらないことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        points = np.array(
            [[0, 0], [10, 0], [10, 10], [0, 10], [2, 7], [8, 3]],
            dtype=np.int32,
        )

        result = smooth_contour(points, 4)

        self.assertEqual(points.shape, result.shape)
        self.assertEqual(np.int32, result.dtype)


if __name__ == "__main__":
    unittest.main()
