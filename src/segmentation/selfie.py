"""MediaPipe Selfie Segmenter の初期化と実行処理を定義する。"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def create_selfie_segmenter(model_path: str):
    """
    MediaPipeの人物セグメンターを初期化する。

    Args:
        model_path:
            モデルファイルのパス

    Returns:
        初期化済みの ImageSegmenter

    Side Effects:
        MediaPipeのセグメンターを作成する。
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def extract_category_mask(frame: np.ndarray, segmenter) -> np.ndarray:
    """
    BGRフレームからMediaPipeのカテゴリマスクを取得する。

    Args:
        frame:
            BGR形式の入力フレーム
        segmenter:
            初期化済みの ImageSegmenter

    Returns:
        MediaPipeが返すカテゴリマスク

    Side Effects:
        MediaPipeの推論を実行する。
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    segmentation_result = segmenter.segment(mp_image)
    return segmentation_result.category_mask.numpy_view()
