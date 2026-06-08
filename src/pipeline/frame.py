"""1フレームを線画フレームへ変換する処理を定義する。"""

import numpy as np

from config import settings


def create_blank_frame(width: int, height: int) -> np.ndarray:
    """
    黒背景の出力フレームを作成する。

    Args:
        width:
            フレーム幅
        height:
            フレーム高さ

    Returns:
        黒背景のBGRフレーム

    Side Effects:
        なし
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def process_frame(
    frame: np.ndarray,
    segmenter,
    width: int,
    height: int,
    sketch_threshold: int,
    num_frequencies: int,
    line_color_bgr: tuple[int, int, int],
    category_mask_extractor=None,
    person_mask_extractor=None,
    silhouette_finder=None,
    strict_mask_creator=None,
    sketch_edge_extractor=None,
    inner_contour_finder=None,
    line_drawer=None,
) -> np.ndarray:
    """
    入力フレームから線画フレームを生成する。

    Args:
        frame:
            BGR形式の入力フレーム
        segmenter:
            初期化済みのセグメンター
        width:
            出力フレーム幅
        height:
            出力フレーム高さ
        sketch_threshold:
            スケッチ線を抽出する閾値
        num_frequencies:
            フーリエ平滑化で残す周波数成分数
        line_color_bgr:
            BGR形式の線色
        category_mask_extractor:
            カテゴリマスクを取得する関数
        person_mask_extractor:
            人物マスクを作成する関数
        silhouette_finder:
            外枠輪郭を抽出する関数
        strict_mask_creator:
            収縮人物マスクを作成する関数
        sketch_edge_extractor:
            スケッチエッジを抽出する関数
        inner_contour_finder:
            内側輪郭を抽出する関数
        line_drawer:
            輪郭を描画する関数

    Returns:
        線画化したBGRフレーム

    Side Effects:
        line_drawer が出力フレームへ線を描画する。
    """
    if category_mask_extractor is None:
        from segmentation.selfie import extract_category_mask

        category_mask_extractor = extract_category_mask
    if person_mask_extractor is None:
        from edge.sketch import extract_person_mask

        person_mask_extractor = extract_person_mask
    if silhouette_finder is None:
        from edge.sketch import find_silhouette_contours

        silhouette_finder = find_silhouette_contours
    if strict_mask_creator is None:
        from edge.sketch import create_strict_person_mask

        strict_mask_creator = create_strict_person_mask
    if sketch_edge_extractor is None:
        from edge.sketch import extract_sketch_edges

        sketch_edge_extractor = extract_sketch_edges
    if inner_contour_finder is None:
        from edge.sketch import find_inner_contours

        inner_contour_finder = find_inner_contours
    if line_drawer is None:
        from render.drawing import draw_fourier_lines

        line_drawer = draw_fourier_lines

    display_frame = create_blank_frame(width, height)

    category_mask = category_mask_extractor(frame, segmenter)
    person_mask = person_mask_extractor(category_mask)
    silhouette_contours = silhouette_finder(person_mask)
    strict_person_mask = strict_mask_creator(person_mask)
    full_edges = sketch_edge_extractor(frame, sketch_threshold)
    inner_contours = inner_contour_finder(full_edges, strict_person_mask)

    line_drawer(
        display_frame,
        silhouette_contours,
        num_frequencies,
        line_color_bgr,
        is_closed=True,
        line_thickness=settings.SILHOUETTE_LINE_THICKNESS,
    )
    line_drawer(
        display_frame,
        inner_contours,
        num_frequencies,
        line_color_bgr,
        is_closed=False,
        line_thickness=settings.INNER_LINE_THICKNESS,
    )

    return display_frame


def process_frame_with_debug(
    frame: np.ndarray,
    segmenter,
    width: int,
    height: int,
    sketch_threshold: int,
    num_frequencies: int,
    line_color_bgr: tuple[int, int, int],
    previous_edge_history: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    入力フレームから線画フレームと比較表示用の中間画像を生成する。

    Args:
        frame:
            BGR形式の入力フレーム
        segmenter:
            初期化済みのセグメンター
        width:
            出力フレーム幅
        height:
            出力フレーム高さ
        sketch_threshold:
            スケッチ線を抽出する閾値
        num_frequencies:
            フーリエ平滑化で残す周波数成分数
        line_color_bgr:
            BGR形式の線色
        previous_edge_history:
            前フレームまでのエッジ履歴

    Returns:
        線画化したBGRフレーム、比較表示用の中間画像、エッジ履歴

    Side Effects:
        出力フレームへ線を描画する。
    """
    from edge.sketch import (
        create_sketch_debug_images,
        create_strict_person_mask,
        extract_person_mask,
        find_inner_contours,
        find_silhouette_contours,
        stabilize_edges_over_time,
    )
    from render.drawing import draw_fourier_lines
    from segmentation.selfie import extract_category_mask

    display_frame = create_blank_frame(width, height)

    category_mask = extract_category_mask(frame, segmenter)
    person_mask = extract_person_mask(category_mask)
    silhouette_contours = find_silhouette_contours(person_mask)
    strict_person_mask = create_strict_person_mask(person_mask)
    sketch_debug_images = create_sketch_debug_images(frame, sketch_threshold)
    stable_edges, edge_history = stabilize_edges_over_time(
        sketch_debug_images["edges"],
        previous_edge_history,
    )
    inner_contours = find_inner_contours(
        stable_edges,
        strict_person_mask,
    )

    draw_fourier_lines(
        display_frame,
        silhouette_contours,
        num_frequencies,
        line_color_bgr,
        is_closed=True,
        line_thickness=settings.SILHOUETTE_LINE_THICKNESS,
    )
    draw_fourier_lines(
        display_frame,
        inner_contours,
        num_frequencies,
        line_color_bgr,
        is_closed=False,
        line_thickness=settings.INNER_LINE_THICKNESS,
    )

    debug_frames = {
        "camera": frame,
        "mask": person_mask,
        "preprocessed": sketch_debug_images["preprocessed"],
        "soft": sketch_debug_images["soft"],
        "edges": stable_edges,
        "fourier": display_frame,
    }
    return display_frame, debug_frames, edge_history
