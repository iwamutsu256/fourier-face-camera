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

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            h, w, _ = frame.shape

            display_frame = np.zeros((h, w, 3), dtype=np.uint8) if drawing_mode else frame.copy()

            # 1. 人物のセグメンテーション（切り抜き）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            segmentation_result = self.segmenter.segment(mp_image)
            
            # category_mask は背景が0、人物が1〜の配列
            category_mask = segmentation_result.category_mask.numpy_view()
            # 人物の部分だけを255(白)にしたバイナリマスクを作成
            person_mask = (category_mask > 0).astype(np.uint8) * 255

            # 2. マスクを使って背景を黒に塗りつぶす
            fg_image = cv2.bitwise_and(frame, frame, mask=person_mask)

            # 3. グレースケール化してCannyエッジ検出
            gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)

            # 4. エッジの画像から「繋がった線の座標リスト（輪郭）」を抽出
            # RETR_LIST は外側の輪郭だけでなく、眼鏡や服のシワなど内側の線も拾う設定
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # ゴミ（短すぎる線）を描画しないように、長さ順にソートして上位20本の線だけを処理する
            contours = sorted(contours, key=len, reverse=True)[:20]

            # 5. 抽出した各エッジに対してフーリエ変換をかける
            for contour in contours:
                pts_raw = contour.reshape(-1, 2)
                
                # 線が短すぎるとFFTがエラーになるためスキップ
                if len(pts_raw) < max(self.num_frequencies, 4):
                    continue

                pts_smooth = self._apply_fourier_smoothing(pts_raw, self.num_frequencies)
                pts_smooth = pts_smooth.reshape((-1, 1, 2))
                pts_raw = pts_raw.reshape((-1, 1, 2))

                # 描画（赤い細線が元のエッジ、太い水色がフーリエ近似線）
                cv2.polylines(display_frame, [pts_raw], isClosed=False, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(display_frame, [pts_smooth], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # UI表示
            cv2.putText(display_frame, f"Freqs: {self.num_frequencies} | Canny: {self.edge_threshold1}-{self.edge_threshold2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "UP/DOWN: Freqs | LEFT/RIGHT: Edge detail | 'd': Bg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Fourier Face Camera", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                drawing_mode = not drawing_mode
            elif key == 0: # 上矢印
                self.num_frequencies = min(self.num_frequencies + 2, 60)
            elif key == 1: # 下矢印
                self.num_frequencies = max(self.num_frequencies - 2, 2)
            elif key == 2: # 左矢印（Canny閾値を上げて線を減らす）
                self.edge_threshold1 = min(self.edge_threshold1 + 10, 200)
                self.edge_threshold2 = min(self.edge_threshold2 + 10, 300)
            elif key == 3: # 右矢印（Canny閾値を下げて線を増やす）
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