import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

class FourierFaceCamera:
    """
    人物を切り抜き、見た目（眼鏡やマスク含む）のエッジをフーリエ線画に変換するクラス
    """
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
            
        self._prepare_model()
        self._init_segmenter()

        # フーリエ変換で残す波の数（少ないほど丸くなる）
        self.num_frequencies = 10
        # Cannyエッジ検出の感度（低いほど細かい線を拾う）
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150

    def _prepare_model(self):
        """人物切り抜き用モデル（Selfie Segmenter）をダウンロード"""
        model_path = 'selfie_segmenter.tflite'
        if not os.path.exists(model_path):
            print("[INFO] Downloading Selfie Segmenter model. Please wait...")
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model download complete.")
        self.model_path = model_path

    def _init_segmenter(self):
        """ImageSegmenterの初期化"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def _apply_fourier_smoothing(self, points: np.ndarray, num_freqs: int) -> np.ndarray:
        """点群(x, y)を複素数化してFFTをかけ、高周波をカットしてIFFTで戻す（変更なし！）"""
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
        print("[INFO] Starting main loop. Press 'q' to exit.")
        drawing_mode = False
        show_mask_mode = False 

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            h, w, _ = frame.shape

            display_frame = np.zeros((h, w, 3), dtype=np.uint8) if drawing_mode else frame.copy()

            # 1. 人物のセグメンテーション
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            segmentation_result = self.segmenter.segment(mp_image)
            
            category_mask = segmentation_result.category_mask.numpy_view()
            person_mask = (category_mask == 0).astype(np.uint8) * 255

            # ---------------------------------------------------------
            # [進化ポイント1] 完璧な外枠（シルエット）の抽出
            # ---------------------------------------------------------
            # 収縮させる前のマスクから、一番外側の輪郭（RETR_EXTERNAL）を取得する
            silhouette_contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 最も長い1本（人物の大枠）だけを取得
            silhouette_contours = sorted(silhouette_contours, key=len, reverse=True)[:1]

            # ---------------------------------------------------------
            # [進化ポイント2] 内部ディテールのノイズ除去と結合
            # ---------------------------------------------------------
            # マスクの収縮（背景ノイズを削る用）
            kernel = np.ones((5, 5), np.uint8)
            strict_person_mask = cv2.erode(person_mask, kernel, iterations=2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ガウシアンブラーで微細なテクスチャ（毛穴や服の繊維）を消す
            blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # エッジ検出
            full_edges = cv2.Canny(blurred_gray, self.edge_threshold1, self.edge_threshold2)

            # クロージング処理で、途切れた線をくっつける（糊付け）
            close_kernel = np.ones((3, 3), np.uint8)
            full_edges = cv2.morphologyEx(full_edges, cv2.MORPH_CLOSE, close_kernel)

            # 人物の内側のエッジだけを残す
            person_edges = cv2.bitwise_and(full_edges, full_edges, mask=strict_person_mask)

            # 内部の輪郭抽出
            inner_contours, _ = cv2.findContours(person_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # 短いゴミを無視し、上位15本ほどの「意味のある長い線」だけを取得
            inner_contours = sorted(inner_contours, key=len, reverse=True)[:15]

            # ---------------------------------------------------------
            # [進化ポイント3] 描画処理の共通化と美化
            # ---------------------------------------------------------
            def draw_fourier_lines(contour_list, is_closed, smooth_thickness):
                """フーリエ平滑化と描画を行うローカル関数"""
                for contour in contour_list:
                    pts_raw = contour.reshape(-1, 2)
                    if len(pts_raw) < max(self.num_frequencies, 4): continue
                    pts_smooth = self._apply_fourier_smoothing(pts_raw, self.num_frequencies)
                    
                    pts_raw_re = pts_raw.reshape((-1, 1, 2))
                    pts_smooth_re = pts_smooth.reshape((-1, 1, 2))
                    
                    # 元の線を細い赤で、フーリエ線を太い水色で描画
                    cv2.polylines(display_frame, [pts_raw_re], isClosed=is_closed, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    cv2.polylines(display_frame, [pts_smooth_re], isClosed=is_closed, color=(255, 255, 0), thickness=smooth_thickness, lineType=cv2.LINE_AA)

            # シルエットは「閉じた線(True)」で「太く(3)」描く
            draw_fourier_lines(silhouette_contours, is_closed=True, smooth_thickness=3)
            # 内部のディテールは「開いた線(False)」で「少し細く(2)」描く
            draw_fourier_lines(inner_contours, is_closed=False, smooth_thickness=2)

            # UI表示
            cv2.putText(display_frame, f"Freqs: {self.num_frequencies} | Canny: {self.edge_threshold1}-{self.edge_threshold2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "UP/DOWN: Freqs | L/R: Edge | 'd': Bg | 'm': Mask", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if show_mask_mode:
                mask_bgr = cv2.cvtColor(strict_person_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_bgr, "DEBUG: Mask Mode (Press 'm' to switch)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Fourier Face Camera", mask_bgr)
            else:
                cv2.imshow("Fourier Face Camera", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('d'): drawing_mode = not drawing_mode
            elif key == ord('m'): show_mask_mode = not show_mask_mode
            elif key == 0: self.num_frequencies = min(self.num_frequencies + 5, 300)
            elif key == 1: self.num_frequencies = max(self.num_frequencies - 5, 2)
            elif key == 2: 
                self.edge_threshold1 = min(self.edge_threshold1 + 10, 200)
                self.edge_threshold2 = min(self.edge_threshold2 + 10, 300)
            elif key == 3: 
                self.edge_threshold1 = max(self.edge_threshold1 - 10, 10)
                self.edge_threshold2 = max(self.edge_threshold2 - 10, 50)

        self.cleanup()

    def cleanup(self) -> None:
        self.cap.release()
        self.segmenter.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed.")

if __name__ == "__main__":
    app = FourierFaceCamera()
    app.run()