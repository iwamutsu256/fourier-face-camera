"""スケッチ抽出前処理の基本動作を確認するテスト。"""

import unittest

import numpy as np

from src.config import settings
from src.edge.sketch import calculate_canny_thresholds, preprocess_gray_for_edges


class EdgeSketchTest(unittest.TestCase):
    """輪郭抽出に使う画像処理を確認する。"""

    def test_preprocess_gray_for_edges_returns_gray_frame(self):
        """
        前処理後の画像が入力の高さ・幅を保つことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :8] = (30, 30, 30)
        frame[:, 8:] = (220, 220, 220)

        result = preprocess_gray_for_edges(frame)

        self.assertEqual((12, 16), result.shape)
        self.assertEqual(np.uint8, result.dtype)

    def test_calculate_canny_thresholds_stays_in_valid_range(self):
        """
        自動Canny閾値が8bit画像として有効な範囲に収まることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        image = np.array([[0, 32], [128, 255]], dtype=np.uint8)

        lower, upper = calculate_canny_thresholds(image)

        self.assertGreaterEqual(lower, 0)
        self.assertLessEqual(upper, settings.SKETCH_MAX_VALUE)
        self.assertLess(lower, upper)


if __name__ == "__main__":
    unittest.main()
