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

    def test_lighting_preprocess_settings_are_valid(self):
        """
        照明補正用の設定値がOpenCVに渡せる範囲であることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        tile_width, tile_height = settings.CLAHE_TILE_GRID_SIZE
        self.assertGreater(settings.CLAHE_CLIP_LIMIT, 0)
        self.assertGreater(tile_width, 0)
        self.assertGreater(tile_height, 0)
        self.assertGreater(settings.BILATERAL_FILTER_DIAMETER, 0)
        self.assertGreater(settings.ADAPTIVE_THRESHOLD_BLOCK_SIZE, 1)
        self.assertEqual(1, settings.ADAPTIVE_THRESHOLD_BLOCK_SIZE % 2)
        self.assertGreater(settings.BINARY_DENOISE_MEDIAN_KERNEL_SIZE, 1)
        self.assertEqual(1, settings.BINARY_DENOISE_MEDIAN_KERNEL_SIZE % 2)
        binary_open_width, binary_open_height = settings.BINARY_OPEN_KERNEL_SIZE
        self.assertGreater(binary_open_width, 0)
        self.assertGreater(binary_open_height, 0)
        self.assertGreaterEqual(settings.BINARY_OPEN_ITERATIONS, 0)
        edge_blur_width, edge_blur_height = settings.EDGE_SOURCE_BLUR_KERNEL_SIZE
        self.assertEqual(1, edge_blur_width % 2)
        self.assertEqual(1, edge_blur_height % 2)
        self.assertGreater(settings.CANNY_AUTO_SIGMA, 0)
        self.assertLess(settings.CANNY_AUTO_SIGMA, 1)
        self.assertGreater(settings.MIN_EDGE_COMPONENT_AREA, 0)
        self.assertGreater(settings.MIN_EDGE_COMPONENT_SIZE, 0)
        self.assertGreater(settings.TEMPORAL_EDGE_DECAY, 0)
        self.assertLess(settings.TEMPORAL_EDGE_DECAY, 1)
        self.assertGreater(settings.TEMPORAL_EDGE_THRESHOLD, 0)
        self.assertLessEqual(settings.TEMPORAL_EDGE_THRESHOLD, settings.SKETCH_MAX_VALUE)
        self.assertGreater(settings.MIN_INNER_CONTOUR_POINTS, 0)

    def test_comparison_tile_size_is_valid(self):
        """
        比較モニターのタイルサイズが正の値であることを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        tile_width, tile_height = settings.COMPARISON_TILE_SIZE
        self.assertGreater(tile_width, 0)
        self.assertGreater(tile_height, 0)


if __name__ == "__main__":
    unittest.main()
