import cv2
import numpy as np
import pyvirtualcam

from config import settings
from edge.sketch import (
    create_strict_person_mask,
    extract_person_mask,
    extract_sketch_edges,
    find_inner_contours,
    find_silhouette_contours,
)
from render.drawing import draw_fourier_lines
from segmentation.model import prepare_model
from segmentation.selfie import create_selfie_segmenter, extract_category_mask

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
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
            
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
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = settings.FALLBACK_FPS

        print(f"[INFO] Starting virtual camera at {w}x{h} ({fps}fps)...")
        
        with pyvirtualcam.Camera(width=w, height=h, fps=fps) as cam:
            print("[INFO] Virtual camera active: OBS Virtual Camera")
            print("[INFO] Press 'q' on the monitor window to exit.")

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                display_frame = np.zeros((h, w, 3), dtype=np.uint8)

                category_mask = extract_category_mask(frame, self.segmenter)
                person_mask = extract_person_mask(category_mask)
                silhouette_contours = find_silhouette_contours(person_mask)
                strict_person_mask = create_strict_person_mask(person_mask)
                full_edges = extract_sketch_edges(frame, self.sketch_threshold)
                inner_contours = find_inner_contours(full_edges, strict_person_mask)

                draw_fourier_lines(
                    display_frame,
                    silhouette_contours,
                    self.num_frequencies,
                    self.line_color_bgr,
                    is_closed=True,
                    line_thickness=settings.SILHOUETTE_LINE_THICKNESS,
                )
                draw_fourier_lines(
                    display_frame,
                    inner_contours,
                    self.num_frequencies,
                    self.line_color_bgr,
                    is_closed=False,
                    line_thickness=settings.INNER_LINE_THICKNESS,
                )

                out_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                cam.send(out_frame_rgb)
                cam.sleep_until_next_frame()

                cv2.imshow("Fourier Face Camera (Monitor)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
        self.cap.release()
        self.segmenter.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed.")

if __name__ == "__main__":
    # line_color_rgb: 線の色
    # sketch_threshold: 数値を小さくする(例:15)と細かい表情を拾い、大きくする(例:45)と主要な線だけになります。
    app = FourierFaceCamera(
        line_color_rgb=settings.APP_LINE_COLOR_RGB,
        sketch_threshold=settings.APP_SKETCH_THRESHOLD,
    )
    app.run()
