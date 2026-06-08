"""フーリエ平滑化処理の基本動作を確認するテスト。"""

import unittest
from pathlib import Path
import sys

import numpy as np

from src.config import settings

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from main import FourierFaceCamera
except ModuleNotFoundError as error:
    if error.name not in {"cv2", "mediapipe", "pyvirtualcam"}:
        raise
    FourierFaceCamera = None
    IMPORT_ERROR_NAME = error.name
else:
    IMPORT_ERROR_NAME = ""


@unittest.skipIf(
    FourierFaceCamera is None,
    f"実行環境に {IMPORT_ERROR_NAME} がないためスキップする。",
)
class FourierSmoothingTest(unittest.TestCase):
    """カメラを初期化せずにフーリエ平滑化を確認する。"""

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

        result = FourierFaceCamera._apply_fourier_smoothing(
            None,
            points,
            settings.DEFAULT_NUM_FREQUENCIES,
        )

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

        result = FourierFaceCamera._apply_fourier_smoothing(None, points, 4)

        self.assertEqual(points.shape, result.shape)
        self.assertEqual(np.int32, result.dtype)


if __name__ == "__main__":
    unittest.main()
