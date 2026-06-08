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
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    _, binary_sketch = cv2.threshold(
        inverted_sketch,
        sketch_threshold,
        settings.SKETCH_MAX_VALUE,
        cv2.THRESH_BINARY,
    )
    edges = cv2.Canny(
        binary_sketch,
        settings.CANNY_LOW_THRESHOLD,
        settings.CANNY_HIGH_THRESHOLD,
    )
    close_kernel = np.ones(settings.CLOSE_KERNEL_SIZE, np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)


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
    return sorted(contours, key=len, reverse=True)[: settings.MAX_INNER_CONTOURS]
