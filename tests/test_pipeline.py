"""フレーム処理パイプラインを確認するテスト。"""

import unittest

import numpy as np

from src.config import settings
from src.pipeline.frame import create_blank_frame, process_frame


class FramePipelineTest(unittest.TestCase):
    """1フレーム処理の流れを確認する。"""

    def test_create_blank_frame_uses_requested_size(self):
        """
        指定した幅と高さで黒背景フレームを作成することを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        frame = create_blank_frame(320, 240)

        self.assertEqual((240, 320, 3), frame.shape)
        self.assertEqual(np.uint8, frame.dtype)
        self.assertEqual(0, int(frame.sum()))

    def test_process_frame_draws_silhouette_and_inner_contours(self):
        """
        外枠輪郭と内側輪郭をそれぞれ描画することを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            なし
        """
        input_frame = np.ones((2, 3, 3), dtype=np.uint8)
        calls = []

        def fake_category_mask_extractor(frame, segmenter):
            calls.append(("category", frame.shape, segmenter))
            return "category-mask"

        def fake_person_mask_extractor(category_mask):
            calls.append(("person", category_mask))
            return "person-mask"

        def fake_silhouette_finder(person_mask):
            calls.append(("silhouette", person_mask))
            return ["silhouette-contour"]

        def fake_strict_mask_creator(person_mask):
            calls.append(("strict", person_mask))
            return "strict-mask"

        def fake_sketch_edge_extractor(frame, sketch_threshold):
            calls.append(("edges", frame.shape, sketch_threshold))
            return "edges"

        def fake_inner_contour_finder(edges, strict_person_mask):
            calls.append(("inner", edges, strict_person_mask))
            return ["inner-contour"]

        def fake_line_drawer(
            display_frame,
            contour_list,
            num_frequencies,
            line_color_bgr,
            is_closed,
            line_thickness,
        ):
            calls.append(
                (
                    "draw",
                    contour_list,
                    num_frequencies,
                    line_color_bgr,
                    is_closed,
                    line_thickness,
                    display_frame.shape,
                )
            )

        result = process_frame(
            input_frame,
            segmenter="segmenter",
            width=3,
            height=2,
            sketch_threshold=60,
            num_frequencies=10,
            line_color_bgr=(255, 0, 255),
            category_mask_extractor=fake_category_mask_extractor,
            person_mask_extractor=fake_person_mask_extractor,
            silhouette_finder=fake_silhouette_finder,
            strict_mask_creator=fake_strict_mask_creator,
            sketch_edge_extractor=fake_sketch_edge_extractor,
            inner_contour_finder=fake_inner_contour_finder,
            line_drawer=fake_line_drawer,
        )

        self.assertEqual((2, 3, 3), result.shape)
        self.assertEqual(
            [
                ("category", (2, 3, 3), "segmenter"),
                ("person", "category-mask"),
                ("silhouette", "person-mask"),
                ("strict", "person-mask"),
                ("edges", (2, 3, 3), 60),
                ("inner", "edges", "strict-mask"),
                (
                    "draw",
                    ["silhouette-contour"],
                    10,
                    (255, 0, 255),
                    True,
                    settings.SILHOUETTE_LINE_THICKNESS,
                    (2, 3, 3),
                ),
                (
                    "draw",
                    ["inner-contour"],
                    10,
                    (255, 0, 255),
                    False,
                    settings.INNER_LINE_THICKNESS,
                    (2, 3, 3),
                ),
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
