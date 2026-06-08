"""OpenCVのモニターウィンドウ表示処理を定義する。"""

from config import settings


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
