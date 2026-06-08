"""pyvirtualcam を使った仮想カメラ出力処理を定義する。"""


def create_virtual_camera(width: int, height: int, fps: int):
    """
    仮想カメラを作成する。

    Args:
        width:
            出力幅
        height:
            出力高さ
        fps:
            出力FPS

    Returns:
        pyvirtualcam の Camera

    Side Effects:
        仮想カメラデバイスを開く。
    """
    import pyvirtualcam

    return pyvirtualcam.Camera(width=width, height=height, fps=fps)


def convert_bgr_to_rgb(frame):
    """
    BGRフレームをRGBフレームに変換する。

    Args:
        frame:
            BGR形式の入力フレーム

    Returns:
        RGB形式のフレーム

    Side Effects:
        なし
    """
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def send_frame(camera, frame) -> None:
    """
    BGRフレームを仮想カメラへ送信する。

    Args:
        camera:
            pyvirtualcam の Camera
        frame:
            BGR形式の入力フレーム

    Returns:
        なし

    Side Effects:
        仮想カメラへフレームを送信し、次フレーム時刻まで待機する。
    """
    camera.send(convert_bgr_to_rgb(frame))
    camera.sleep_until_next_frame()
