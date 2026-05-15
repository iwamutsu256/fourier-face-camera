import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import pyvirtualcam

class FourierFaceCamera:
    """
    人物を切り抜き、覆い焼きカラーによるスケッチ抽出を経て、フーリエ線画を仮想カメラに出力する本番用クラス
    """
    # line_color_rgb: 線の色。デフォルトはシアン (0, 255, 255)
    # sketch_threshold: 顔のパーツを拾う感度。下げる(例:15)と細かいシワまで拾い、上げる(例:50)と主要な線だけになる
    def __init__(self, camera_id: int = 0, line_color_rgb: tuple = (0, 255, 255), sketch_threshold: int = 30):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
            
        self._prepare_model()
        self._init_segmenter()

        # 本番仕様のパラメーター
        self.num_frequencies = 60
        self.sketch_threshold = sketch_threshold
        
        self.line_color_bgr = (line_color_rgb[2], line_color_rgb[1], line_color_rgb[0])
        print(f"[INFO] Initialized with sketch threshold: {self.sketch_threshold}")

    def _prepare_model(self):
        model_path = 'selfie_segmenter.tflite'
        if not os.path.exists(model_path):
            print("[INFO] Downloading Selfie Segmenter model. Please wait...")
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model download complete.")
        self.model_path = model_path

    def _init_segmenter(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def _apply_fourier_smoothing(self, points: np.ndarray, num_freqs: int) -> np.ndarray:
        if len(points) <= num_freqs:
            return points
        complex_pts = points[:, 0] + 1j * points[:, 1]
        fft_coeffs = np.fft.fft(complex_pts)
        fft_coeffs_filtered = np.zeros_like(fft_coeffs)
        half = num_freqs // 2
        fft_coeffs_filtered[:half] = fft_coeffs[:half]
        fft_coeffs_filtered[-half:] = fft_coeffs[-half:]
        smoothed_complex = np.fft.ifft(fft_coeffs_filtered)
        smoothed_points = np.column_stack((np.real(smoothed_complex), np.imag(smoothed_complex)))
        return smoothed_points.astype(np.int32)

    def run(self) -> None:
        ret, frame = self.cap.read()
        if not ret: return
        h, w, _ = frame.shape
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30

        print(f"[INFO] Starting virtual camera at {w}x{h} ({fps}fps)...")
        
        with pyvirtualcam.Camera(width=w, height=h, fps=fps) as cam:
            print("[INFO] Virtual camera active: OBS Virtual Camera")
            print("[INFO] Press 'q' on the monitor window to exit.")

            while True:
                ret, frame = self.cap.read()
                if not ret: break

                display_frame = np.zeros((h, w, 3), dtype=np.uint8)

                # 1. 人物のセグメンテーション（外枠用）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                segmentation_result = self.segmenter.segment(mp_image)
                category_mask = segmentation_result.category_mask.numpy_view()
                person_mask = (category_mask == 0).astype(np.uint8) * 255

                silhouette_contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                silhouette_contours = sorted(silhouette_contours, key=len, reverse=True)[:1]

                kernel = np.ones((5, 5), np.uint8)
                strict_person_mask = cv2.erode(person_mask, kernel, iterations=2)

                # ---------------------------------------------------------
                # [完全新規] あなたのアイデアを再現した「覆い焼きカラー」線画抽出
                # ---------------------------------------------------------
                # ① 白黒に変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # ② 画像を反転
                inv_gray = cv2.bitwise_not(gray)
                
                # ③ 反転画像を大きくぼかす
                blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
                
                # ④ 元のグレースケールと、ぼかした反転画像を「覆い焼きカラー」でブレンド
                # (スケッチのような白背景に黒い陰影の画像になる)
                sketch = cv2.divide(gray, 255 - blurred, scale=256)
                
                # ⑤ 白黒を再反転（黒背景に白い線にする）
                sketch_inv = cv2.bitwise_not(sketch)
                
                # ⑥ 閾値を設定して線を濃くする（薄いノイズを消す）
                _, binary_sketch = cv2.threshold(sketch_inv, self.sketch_threshold, 255, cv2.THRESH_BINARY)
                
                # ⑦ 抽出された「太い線」から、findContoursに通すための「1ピクセルの輪郭」を抽出
                full_edges = cv2.Canny(binary_sketch, 100, 200)

                # 途切れた線を糊付け
                close_kernel = np.ones((3, 3), np.uint8)
                full_edges = cv2.morphologyEx(full_edges, cv2.MORPH_CLOSE, close_kernel)
                # ---------------------------------------------------------

                # マスクで人物の内側だけに限定
                person_edges = cv2.bitwise_and(full_edges, full_edges, mask=strict_person_mask)
                inner_contours, _ = cv2.findContours(person_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                inner_contours = sorted(inner_contours, key=len, reverse=True)[:25] # パーツが増えるので抽出数を15->25に増加

                # 描画処理
                def draw_fourier_lines(contour_list, is_closed, smooth_thickness):
                    for contour in contour_list:
                        pts_raw = contour.reshape(-1, 2)
                        if len(pts_raw) < max(self.num_frequencies, 4): continue
                        pts_smooth = self._apply_fourier_smoothing(pts_raw, self.num_frequencies)
                        pts_smooth_re = pts_smooth.reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts_smooth_re], isClosed=is_closed, color=self.line_color_bgr, thickness=smooth_thickness, lineType=cv2.LINE_AA)

                draw_fourier_lines(silhouette_contours, is_closed=True, smooth_thickness=3)
                draw_fourier_lines(inner_contours, is_closed=False, smooth_thickness=2)

                out_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                cam.send(out_frame_rgb)
                cam.sleep_until_next_frame()

                cv2.imshow("Fourier Face Camera (Monitor)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cleanup()

    def cleanup(self) -> None:
        self.cap.release()
        self.segmenter.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed.")

if __name__ == "__main__":
    # line_color_rgb: 線の色
    # sketch_threshold: 数値を小さくする(例:15)と細かい表情を拾い、大きくする(例:45)と主要な線だけになります。
    app = FourierFaceCamera(line_color_rgb=(0, 255, 255), sketch_threshold=60)
    app.run()