"""人物領域とスケッチ風エッジを抽出する処理を定義する。"""

import cv2
import numpy as np

from config import settings


def extract_person_mask(category_mask: np.ndarray) -> np.ndarray:
    """
    MediaPipeのカテゴリマスクから人物マスクを作成する。

    Args:
        category_mask:
            MediaPipeが返すカテゴリマスク

    Returns:
        人物領域を255、それ以外を0にしたマスク

    Side Effects:
        なし
    """
    return (category_mask == 0).astype(np.uint8) * settings.SKETCH_MAX_VALUE


def find_silhouette_contours(person_mask: np.ndarray) -> list[np.ndarray]:
    """
    人物マスクから外枠輪郭を抽出する。

    Args:
        person_mask:
            人物領域マスク

    Returns:
        長い順に絞り込んだ外枠輪郭リスト

    Side Effects:
        なし
    """
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sorted(contours, key=len, reverse=True)[:1]


def create_strict_person_mask(person_mask: np.ndarray) -> np.ndarray:
    """
    内側線の抽出に使うため、人物マスクを少し縮小する。

    Args:
        person_mask:
            人物領域マスク

    Returns:
        収縮後の人物領域マスク

    Side Effects:
        なし
    """
    kernel = np.ones(settings.ERODE_KERNEL_SIZE, np.uint8)
    return cv2.erode(person_mask, kernel, iterations=settings.ERODE_ITERATIONS)


def preprocess_gray_for_edges(frame: np.ndarray) -> np.ndarray:
    """
    照明ムラの影響を抑え、輪郭抽出に使いやすいグレースケール画像へ変換する。

    Args:
        frame:
            BGR形式の入力フレーム

    Returns:
        コントラスト補正とノイズ低減後のグレースケール画像

    Side Effects:
        なし
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=settings.CLAHE_CLIP_LIMIT,
        tileGridSize=settings.CLAHE_TILE_GRID_SIZE,
    )
    equalized_gray = clahe.apply(gray_frame)
    return cv2.bilateralFilter(
        equalized_gray,
        settings.BILATERAL_FILTER_DIAMETER,
        settings.BILATERAL_FILTER_SIGMA_COLOR,
        settings.BILATERAL_FILTER_SIGMA_SPACE,
    )


def calculate_canny_thresholds(image: np.ndarray) -> tuple[int, int]:
    """
    画像の明るさ分布からCannyの閾値を計算する。

    Args:
        image:
            8bitグレースケール画像

    Returns:
        Cannyの下限・上限閾値

    Side Effects:
        なし
    """
    median_value = float(np.median(image))
    sigma = settings.CANNY_AUTO_SIGMA
    lower = int(max(0, (1.0 - sigma) * median_value))
    upper = int(min(settings.SKETCH_MAX_VALUE, (1.0 + sigma) * median_value))
    return lower, max(lower + 1, upper)


def create_soft_edge_source(inverted_sketch: np.ndarray) -> np.ndarray:
    """
    2値化せずに、Cannyへ渡すための中間グレースケール画像を作成する。

    Args:
        inverted_sketch:
            線を明るくしたグレースケールのスケッチ画像

    Returns:
        ノイズを抑えた中間グレースケール画像

    Side Effects:
        なし
    """
    denoised_sketch = cv2.medianBlur(
        inverted_sketch,
        settings.BINARY_DENOISE_MEDIAN_KERNEL_SIZE,
    )
    return cv2.GaussianBlur(
        denoised_sketch,
        settings.EDGE_SOURCE_BLUR_KERNEL_SIZE,
        0,
    )


def binarize_sketch(inverted_sketch: np.ndarray, sketch_threshold: int) -> np.ndarray:
    """
    スケッチ画像を輪郭抽出向けの2値画像へ変換する。

    Args:
        inverted_sketch:
            線を明るくしたグレースケールのスケッチ画像
        sketch_threshold:
            固定閾値を使う場合の閾値

    Returns:
        線領域を255、それ以外を0にした2値画像

    Side Effects:
        なし
    """
    denoised_sketch = create_soft_edge_source(inverted_sketch)
    if settings.USE_ADAPTIVE_SKETCH_THRESHOLD:
        binary_sketch = cv2.adaptiveThreshold(
            denoised_sketch,
            settings.SKETCH_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            settings.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            settings.ADAPTIVE_THRESHOLD_C,
        )
    else:
        _, binary_sketch = cv2.threshold(
            denoised_sketch,
            sketch_threshold,
            settings.SKETCH_MAX_VALUE,
            cv2.THRESH_BINARY,
        )

    open_kernel = np.ones(settings.BINARY_OPEN_KERNEL_SIZE, np.uint8)
    return cv2.morphologyEx(
        binary_sketch,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=settings.BINARY_OPEN_ITERATIONS,
    )


def remove_small_edge_components(edges: np.ndarray) -> np.ndarray:
    """
    小さい粒状ノイズを連結成分のサイズで除去する。

    Args:
        edges:
            Canny後のエッジ画像

    Returns:
        小さい連結成分を取り除いたエッジ画像

    Side Effects:
        なし
    """
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges,
        connectivity=8,
    )
    filtered_edges = np.zeros_like(edges)
    for component_index in range(1, component_count):
        area = stats[component_index, cv2.CC_STAT_AREA]
        width = stats[component_index, cv2.CC_STAT_WIDTH]
        height = stats[component_index, cv2.CC_STAT_HEIGHT]
        max_size = max(width, height)
        if (
            area >= settings.MIN_EDGE_COMPONENT_AREA
            and max_size >= settings.MIN_EDGE_COMPONENT_SIZE
        ):
            filtered_edges[labels == component_index] = settings.SKETCH_MAX_VALUE

    return filtered_edges


def stabilize_edges_over_time(
    edges: np.ndarray,
    previous_edge_history: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    前フレームのエッジを短時間だけ残し、線の点滅を抑える。

    Args:
        edges:
            現在フレームのエッジ画像
        previous_edge_history:
            前フレームまでのエッジ履歴

    Returns:
        安定化したエッジ画像と次フレームへ渡すエッジ履歴

    Side Effects:
        なし
    """
    if (
        not settings.USE_TEMPORAL_EDGE_STABILIZATION
        or previous_edge_history is None
        or previous_edge_history.shape != edges.shape
    ):
        return edges, edges.copy()

    decayed_history = (previous_edge_history * settings.TEMPORAL_EDGE_DECAY).astype(
        np.uint8
    )
    edge_history = np.maximum(edges, decayed_history)
    stabilized_edges = np.where(
        edge_history >= settings.TEMPORAL_EDGE_THRESHOLD,
        settings.SKETCH_MAX_VALUE,
        0,
    ).astype(np.uint8)
    return stabilized_edges, edge_history


def create_sketch_debug_images(
    frame: np.ndarray,
    sketch_threshold: int,
) -> dict[str, np.ndarray]:
    """
    スケッチ線抽出の中間画像を作成する。

    Args:
        frame:
            BGR形式の入力フレーム
        sketch_threshold:
            スケッチ線を抽出する閾値

    Returns:
        前処理、2値化、エッジ抽出などの中間画像

    Side Effects:
        なし
    """
    gray_frame = preprocess_gray_for_edges(frame)
    inverted_gray = cv2.bitwise_not(gray_frame)
    blurred_inverted_gray = cv2.GaussianBlur(
        inverted_gray,
        settings.GAUSSIAN_BLUR_KERNEL_SIZE,
        0,
    )
    sketch = cv2.divide(
        gray_frame,
        settings.SKETCH_MAX_VALUE - blurred_inverted_gray,
        scale=settings.SKETCH_DIVIDE_SCALE,
    )
    inverted_sketch = cv2.bitwise_not(sketch)
    soft_edges_source = create_soft_edge_source(inverted_sketch)
    binary_sketch = binarize_sketch(inverted_sketch, sketch_threshold)
    if settings.USE_AUTO_CANNY_THRESHOLDS:
        canny_low_threshold, canny_high_threshold = calculate_canny_thresholds(
            soft_edges_source
        )
    else:
        canny_low_threshold = settings.CANNY_LOW_THRESHOLD
        canny_high_threshold = settings.CANNY_HIGH_THRESHOLD

    edges = cv2.Canny(soft_edges_source, canny_low_threshold, canny_high_threshold)
    close_kernel = np.ones(settings.CLOSE_KERNEL_SIZE, np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    filtered_edges = remove_small_edge_components(closed_edges)
    return {
        "preprocessed": gray_frame,
        "sketch": inverted_sketch,
        "soft": soft_edges_source,
        "binary": binary_sketch,
        "edges": filtered_edges,
    }


def extract_sketch_edges(frame: np.ndarray, sketch_threshold: int) -> np.ndarray:
    """
    覆い焼きカラー風の処理でスケッチ線を抽出する。

    Args:
        frame:
            BGR形式の入力フレーム
        sketch_threshold:
            スケッチ線を抽出する閾値

    Returns:
        CannyとClosingを適用したエッジ画像

    Side Effects:
        なし
    """
    return create_sketch_debug_images(frame, sketch_threshold)["edges"]


def find_inner_contours(edges: np.ndarray, strict_person_mask: np.ndarray) -> list[np.ndarray]:
    """
    人物内側に限定したスケッチ線の輪郭を抽出する。

    Args:
        edges:
            スケッチ線のエッジ画像
        strict_person_mask:
            収縮後の人物領域マスク

    Returns:
        長い順に絞り込んだ内側輪郭リスト

    Side Effects:
        なし
    """
    person_edges = cv2.bitwise_and(edges, edges, mask=strict_person_mask)
    contours, _ = cv2.findContours(person_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [
        contour
        for contour in contours
        if len(contour) >= settings.MIN_INNER_CONTOUR_POINTS
    ]
    return sorted(filtered_contours, key=len, reverse=True)[: settings.MAX_INNER_CONTOURS]
