from capture.camera import (
    get_capture_fps,
    get_frame_size,
    open_camera,
    read_frame,
    release_camera,
)
from config import settings
from display.window import destroy_monitor_windows, show_monitor_window
from pipeline.frame import process_frame
from segmentation.model import prepare_model
from segmentation.selfie import create_selfie_segmenter
from virtual_camera.output import create_virtual_camera, send_frame

class FourierFaceCamera:
    """
    人物を切り抜き、覆い焼きカラーによるスケッチ抽出を経て、フーリエ線画を仮想カメラに出力する本番用クラス
    """
    # line_color_rgb: 線の色。デフォルトはシアン (0, 255, 255)
    # sketch_threshold: 顔のパーツを拾う感度。下げる(例:15)と細かいシワまで拾い、上げる(例:50)と主要な線だけになる
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

        # 本番仕様のパラメーター
        self.num_frequencies = settings.DEFAULT_NUM_FREQUENCIES
        self.sketch_threshold = sketch_threshold
        
        self.line_color_bgr = (line_color_rgb[2], line_color_rgb[1], line_color_rgb[0])
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

            while True:
                ret, frame = read_frame(self.cap)
                if not ret:
                    break

                display_frame = process_frame(
                    frame,
                    self.segmenter,
                    width,
                    height,
                    self.sketch_threshold,
                    self.num_frequencies,
                    self.line_color_bgr,
                )

                send_frame(cam, display_frame)

                if show_monitor_window(display_frame):
                    break

        self.cleanup()

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

if __name__ == "__main__":
    # line_color_rgb: 線の色
    # sketch_threshold: 数値を小さくする(例:15)と細かい表情を拾い、大きくする(例:45)と主要な線だけになります。
    app = FourierFaceCamera(
        line_color_rgb=settings.APP_LINE_COLOR_RGB,
        sketch_threshold=settings.APP_SKETCH_THRESHOLD,
    )
    app.run()
