"""輪郭点を線画として描画する処理を定義する。"""

import cv2

from config import settings
from fourier.smoothing import smooth_contour


def draw_fourier_lines(
    display_frame,
    contour_list,
    num_frequencies: int,
    line_color_bgr: tuple[int, int, int],
    is_closed: bool,
    line_thickness: int,
) -> None:
    """
    輪郭点列をフーリエ平滑化して描画する。

    Args:
        display_frame:
            描画先フレーム
        contour_list:
            描画対象の輪郭リスト
        num_frequencies:
            残す周波数成分数
        line_color_bgr:
            BGR形式の線色
        is_closed:
            閉じた線として描画するかどうか
        line_thickness:
            描画する線の太さ

    Returns:
        なし

    Side Effects:
        display_frame に線を描画する。
    """
    for contour in contour_list:
        raw_points = contour.reshape(-1, 2)
        if len(raw_points) < max(num_frequencies, settings.MIN_CONTOUR_POINTS):
            continue

        smoothed_points = smooth_contour(raw_points, num_frequencies)
        reshaped_points = smoothed_points.reshape((-1, 1, 2))
        cv2.polylines(
            display_frame,
            [reshaped_points],
            isClosed=is_closed,
            color=line_color_bgr,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
