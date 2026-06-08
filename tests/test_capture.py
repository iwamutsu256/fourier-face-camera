"""カメラ入力の補助処理を確認するテスト。"""

import unittest
from pathlib import Path
import sys

import numpy as np

from src.config import settings

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from capture.camera import get_frame_size, resolve_fps


class CaptureTest(unittest.TestCase):
    """カメラ入力の補助関数を確認する。"""

    def test_get_frame_size_returns_width_and_height(self):
        """
        フレーム形状から幅と高さを取得できることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        width, height = get_frame_size(frame)

        self.assertEqual(640, width)
        self.assertEqual(480, height)

    def test_resolve_fps_uses_fallback_when_fps_is_zero(self):
        """
        FPSが0の場合にfallback値を使うことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        self.assertEqual(settings.FALLBACK_FPS, resolve_fps(0))

    def test_resolve_fps_keeps_camera_fps(self):
        """
        FPSが取得できている場合にその値を使うことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        self.assertEqual(60, resolve_fps(60))


if __name__ == "__main__":
    unittest.main()
