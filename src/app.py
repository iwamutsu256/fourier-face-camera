"""Fourier Face Camera アプリケーション本体を定義する。"""

from capture.camera import (
    get_capture_fps,
    get_frame_size,
    open_camera,
    read_frame,
    release_camera,
)
from config import settings
from display.window import (
    create_comparison_view,
    destroy_monitor_windows,
    show_monitor_window,
)
from pipeline.frame import process_frame, process_frame_with_debug
from segmentation.model import prepare_model
from segmentation.selfie import create_selfie_segmenter
from virtual_camera.output import create_virtual_camera, send_frame


class FourierFaceCamera:
    """
    人物を切り抜き、スケッチ抽出とフーリエ線画化を経て仮想カメラに出力する。
    """

    def __init__(
        self,
        camera_id: int = settings.DEFAULT_CAMERA_ID,
        line_color_rgb: tuple = settings.DEFAULT_LINE_COLOR_RGB,
        sketch_threshold: int = settings.DEFAULT_SKETCH_THRESHOLD,
    ):
        """
        カメラ、セグメンテーション、描画設定を初期化する。

        Args:
            camera_id:
                使用するカメラID
            line_color_rgb:
                線の色
            sketch_threshold:
                スケッチ線を抽出する閾値

        Returns:
            なし

        Side Effects:
            カメラとMediaPipeセグメンターを初期化する。
        """
        self.cap = open_camera(camera_id)
        self.model_path = prepare_model()
        self.segmenter = create_selfie_segmenter(self.model_path)
        self.num_frequencies = settings.DEFAULT_NUM_FREQUENCIES
        self.sketch_threshold = sketch_threshold
        self.line_color_bgr = (line_color_rgb[2], line_color_rgb[1], line_color_rgb[0])
        self.edge_history = None
        print(f"[INFO] Initialized with sketch threshold: {self.sketch_threshold}")

    def run(self) -> None:
        """
        カメラ映像を線画化し、仮想カメラへ送信する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            カメラ入力、画面表示、仮想カメラ出力を行う。
        """
        ret, frame = read_frame(self.cap)
        if not ret:
            return

        width, height = get_frame_size(frame)
        fps = get_capture_fps(self.cap)

        print(f"[INFO] Starting virtual camera at {width}x{height} ({fps}fps)...")
        with create_virtual_camera(width=width, height=height, fps=fps) as cam:
            print("[INFO] Virtual camera active: OBS Virtual Camera")
            print("[INFO] Press 'q' on the monitor window to exit.")
            self._run_frame_loop(cam, width, height)

        self.cleanup()

    def _run_frame_loop(self, cam, width: int, height: int) -> None:
        """
        カメラからフレームを読み込み、線画化して出力し続ける。

        Args:
            cam:
                仮想カメラ
            width:
                出力フレーム幅
            height:
                出力フレーム高さ

        Returns:
            なし

        Side Effects:
            仮想カメラ出力とモニター表示を行う。
        """
        while True:
            ret, frame = read_frame(self.cap)
            if not ret:
                break

            if settings.APP_SHOW_COMPARISON_MONITOR:
                (
                    display_frame,
                    debug_frames,
                    self.edge_history,
                ) = process_frame_with_debug(
                    frame,
                    self.segmenter,
                    width,
                    height,
                    self.sketch_threshold,
                    self.num_frequencies,
                    self.line_color_bgr,
                    self.edge_history,
                )
                monitor_frame = create_comparison_view(debug_frames)
            else:
                display_frame = process_frame(
                    frame,
                    self.segmenter,
                    width,
                    height,
                    self.sketch_threshold,
                    self.num_frequencies,
                    self.line_color_bgr,
                )
                monitor_frame = display_frame

            send_frame(cam, display_frame)

            if show_monitor_window(monitor_frame):
                break

    def cleanup(self) -> None:
        """
        カメラ、セグメンター、表示ウィンドウを解放する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            外部リソースを解放する。
        """
        release_camera(self.cap)
        self.segmenter.close()
        destroy_monitor_windows()
        print("[INFO] Camera released and windows destroyed.")


def create_default_app() -> FourierFaceCamera:
    """
    設定値を使ってアプリケーションを作成する。

    Args:
        なし

    Returns:
        初期化済みのFourierFaceCamera

    Side Effects:
        カメラとMediaPipeセグメンターを初期化する。
    """
    return FourierFaceCamera(
        line_color_rgb=settings.APP_LINE_COLOR_RGB,
        sketch_threshold=settings.APP_SKETCH_THRESHOLD,
    )
