"""OpenCVのモニターウィンドウ表示処理を定義する。"""

import numpy as np

from config import settings


def create_comparison_view(frames: dict[str, np.ndarray]) -> np.ndarray:
    """
    処理段階を同時に確認するための比較ビューを作成する。

    Args:
        frames:
            比較表示するフレーム辞書

    Returns:
        2行3列に並べたBGR比較フレーム

    Side Effects:
        なし
    """
    import cv2

    panels = [
        ("Camera", frames["camera"]),
        ("Mask", frames["mask"]),
        ("Preprocessed", frames["preprocessed"]),
        ("Soft", frames["soft"]),
        ("Edges", frames["edges"]),
        ("Fourier", frames["fourier"]),
    ]
    tiles = [_create_labeled_tile(label, frame) for label, frame in panels]
    top_row = np.hstack(tiles[:3])
    bottom_row = np.hstack(tiles[3:])
    return np.vstack([top_row, bottom_row])


def _create_labeled_tile(label: str, frame: np.ndarray) -> np.ndarray:
    """
    比較ビュー用にフレームをリサイズし、ラベルを付ける。

    Args:
        label:
            表示ラベル
        frame:
            表示するフレーム

    Returns:
        ラベル付きBGRフレーム

    Side Effects:
        なし
    """
    import cv2

    tile_width, tile_height = settings.COMPARISON_TILE_SIZE
    tile = _to_bgr_frame(frame)
    interpolation = cv2.INTER_NEAREST if label in {"Mask", "Edges"} else cv2.INTER_AREA
    tile = cv2.resize(tile, (tile_width, tile_height), interpolation=interpolation)
    cv2.rectangle(tile, (0, 0), (tile_width, 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _to_bgr_frame(frame: np.ndarray) -> np.ndarray:
    """
    比較ビューで結合できるようにフレームを3チャンネルBGRへ揃える。

    Args:
        frame:
            表示するフレーム

    Returns:
        3チャンネルBGRフレーム

    Side Effects:
        なし
    """
    import cv2

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def show_monitor_window(frame) -> bool:
    """
    モニターウィンドウにフレームを表示し、終了操作を確認する。

    Args:
        frame:
            表示するフレーム

    Returns:
        終了キーが押された場合はTrue

    Side Effects:
        OpenCVウィンドウへフレームを表示する。
    """
    import cv2

    cv2.imshow(settings.MONITOR_WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF == ord(settings.EXIT_KEY)


def destroy_monitor_windows() -> None:
    """
    OpenCVの表示ウィンドウを破棄する。

    Args:
        なし

    Returns:
        なし

    Side Effects:
        OpenCVウィンドウをすべて閉じる。
    """
    import cv2

    cv2.destroyAllWindows()
