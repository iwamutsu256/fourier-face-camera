"""設定値の基本的な整合性を確認するテスト。"""

import unittest

from src.config import settings


class SettingsTest(unittest.TestCase):
    """設定値の型と範囲を確認する。"""

    def test_line_color_is_rgb_tuple(self):
        """
        線色がRGBの3要素タプルであることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        self.assertEqual(3, len(settings.APP_LINE_COLOR_RGB))
        for color_value in settings.APP_LINE_COLOR_RGB:
            self.assertGreaterEqual(color_value, 0)
            self.assertLessEqual(color_value, settings.SKETCH_MAX_VALUE)

    def test_edge_threshold_order(self):
        """
        Cannyの下限閾値が上限閾値以下であることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        self.assertLessEqual(settings.CANNY_LOW_THRESHOLD, settings.CANNY_HIGH_THRESHOLD)

    def test_kernel_sizes_are_odd(self):
        """
        ぼかしに使うカーネルサイズが奇数であることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        blur_width, blur_height = settings.GAUSSIAN_BLUR_KERNEL_SIZE
        self.assertEqual(1, blur_width % 2)
        self.assertEqual(1, blur_height % 2)


if __name__ == "__main__":
    unittest.main()
