"""OpenCVのカメラ入力を扱う処理を定義する。"""

from config import settings


def open_camera(camera_id: int):
    """
    カメラを開く。

    Args:
        camera_id:
            使用するカメラID

    Returns:
        OpenCVのVideoCapture

    Side Effects:
        カメラデバイスを開く。
    """
    import cv2

    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
    return capture


def get_frame_size(frame) -> tuple[int, int]:
    """
    フレームから幅と高さを取得する。

    Args:
        frame:
            入力フレーム

    Returns:
        幅と高さ

    Side Effects:
        なし
    """
    height, width, _ = frame.shape
    return width, height


def get_capture_fps(capture) -> int:
    """
    カメラのFPSを取得する。

    Args:
        capture:
            OpenCVのVideoCapture

    Returns:
        カメラFPS。取得できない場合は設定値のFPS

    Side Effects:
        なし
    """
    import cv2

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    return resolve_fps(fps)


def resolve_fps(fps: int) -> int:
    """
    カメラから取得したFPSを実行に使うFPSへ変換する。

    Args:
        fps:
            カメラから取得したFPS

    Returns:
        実行に使うFPS

    Side Effects:
        なし
    """
    if fps == 0:
        return settings.FALLBACK_FPS
    return fps


def read_frame(capture):
    """
    カメラから1フレーム読み込む。

    Args:
        capture:
            OpenCVのVideoCapture

    Returns:
        読み込み成否とフレーム

    Side Effects:
        カメラからフレームを取得する。
    """
    return capture.read()


def release_camera(capture) -> None:
    """
    カメラを解放する。

    Args:
        capture:
            OpenCVのVideoCapture

    Returns:
        なし

    Side Effects:
        カメラデバイスを解放する。
    """
    capture.release()
