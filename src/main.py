import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

class FourierFaceCamera:
    """
    リアルタイム映像を取得し、フーリエ変換によるエフェクトをかけて仮想カメラに出力するクラス
    """
    def __init__(self, camera_id: int = 0):
        # cv2.VideoCaptureはOSのカメラAPIを叩き、デバイスと接続するクラス
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
            
        # 1. モデルの準備と初期化
        self._prepare_model()
        self._init_face_landmarker()

        # 2. 顔の輪郭（Face Oval）のエッジ定義をハードコード
        # API変更で定数が消えても動くように、不変の接続インデックスを直接指定
        face_oval_edges = [
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
            (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
            (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
            (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
            (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
            (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
            (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
        ]
        
        # エッジの集合を一筆書きの頂点リストに変換
        self.contour_indices = self._get_continuous_contour(face_oval_edges)

        # [新規追加] フーリエ変換で残す波の数（少ないほど丸くなり、多いほど元の顔に近づく）
        self.num_frequencies = 10

        print(f"[INFO] Contour initialized with {len(self.contour_indices)} points.")
    
    def _prepare_model(self):
        """MediaPipeの推論モデル(.task)が存在しない場合は自動ダウンロードする"""
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("[INFO] Downloading FaceLandmarker model. Please wait...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model download complete.")
        self.model_path = model_path

    def _init_face_landmarker(self):
        """Tasks APIを用いたFaceLandmarkerの初期化"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _get_continuous_contour(self, edge_list: list) -> list[int]:
        """
        エッジの集合 (start, end) から、一筆書きになるように頂点インデックスを並び替える。
        アルゴリズム的には、次数2のグラフの辺を辿ってパスを構築する処理。
        """
        conn_list = edge_list.copy()
        if not conn_list:
            return []

        # 最初の辺を起点にする
        ordered_nodes = []
        current_edge = conn_list.pop(0)
        ordered_nodes.extend([current_edge[0], current_edge[1]])

        # 繋がる辺を探してリストを成長させる
        while conn_list:
            last_node = ordered_nodes[-1]
            found = False
            for i, edge in enumerate(conn_list):
                if edge[0] == last_node:
                    ordered_nodes.append(edge[1])
                    conn_list.pop(i)
                    found = True
                    break
                elif edge[1] == last_node:
                    ordered_nodes.append(edge[0])
                    conn_list.pop(i)
                    found = True
                    break
            
            # もし途切れた場合（通常は閉路なので起きないが安全のため）
            if not found:
                break
                
        return ordered_nodes

    # ---------------------------------------------------------
    # [新規追加] フーリエ変換による平滑化処理
    # ---------------------------------------------------------
    def _apply_fourier_smoothing(self, points: np.ndarray, num_freqs: int) -> np.ndarray:
        """点群(x, y)を複素数化してFFTをかけ、高周波をカットしてIFFTで戻す"""
        if len(points) <= num_freqs:
            return points

        # 1. 座標を複素数 (x + iy) に変換
        complex_pts = points[:, 0] + 1j * points[:, 1]
        
        # 2. 高速フーリエ変換 (FFT)
        fft_coeffs = np.fft.fft(complex_pts)
        
        # 3. 高周波成分を除去する（ゼロ埋めした配列を用意）
        fft_coeffs_filtered = np.zeros_like(fft_coeffs)
        
        # 低周波成分（前半部分と後半部分）だけを元の係数からコピーして残す
        half = num_freqs // 2
        fft_coeffs_filtered[:half] = fft_coeffs[:half]
        fft_coeffs_filtered[-half:] = fft_coeffs[-half:]
        
        # 4. 逆高速フーリエ変換 (IFFT) で座標空間に戻す
        smoothed_complex = np.fft.ifft(fft_coeffs_filtered)
        
        # 5. 実部と虚部を分離して (x, y) の配列に再構築
        smoothed_points = np.column_stack((np.real(smoothed_complex), np.imag(smoothed_complex)))
        return smoothed_points.astype(np.int32)

    def run(self) -> None:
        """
        メイン処理ループ
        """
        print("[INFO] Starting main loop. Press 'q' to exit.")
        # [新規追加] より滑らかな曲線を描くために、描画用の黒いキャンバスを用意する設定
        drawing_mode = False
        while True:
            # ret: フレームの取得に成功したか (bool)
            # frame: 取得した画像データ (np.ndarray)
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break
            h, w, _ = frame.shape

            # カメラ映像をそのまま出すか、黒背景にするか
            display_frame = np.zeros((h, w, 3), dtype=np.uint8) if drawing_mode else frame.copy()

            # 1. BGRからRGBへの変換（MediaPipeの入力要件）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # 2. 推論の実行
            detection_result = self.detector.detect(mp_image)


            # 3. ランドマークが検出された場合の処理
            if detection_result.face_landmarks:
                # 最初の顔のランドマークを取得
                landmarks = detection_result.face_landmarks[0]
                
                # 一筆書き順に座標を取得し、ピクセル空間にマッピング
                contour_points = []
                for idx in self.contour_indices:
                    landmark = landmarks[idx]
                    # 正規化座標 (0.0~1.0) を実ピクセルに変換
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    contour_points.append([px, py])
                
                # 生の点群配列
                pts_raw = np.array(contour_points, np.int32)
                
                # [新規追加] フーリエ変換による平滑化点群の生成
                pts_smooth = self._apply_fourier_smoothing(pts_raw, self.num_frequencies)
                pts_smooth = pts_smooth.reshape((-1, 1, 2))
                pts_raw = pts_raw.reshape((-1, 1, 2))

            # 描画：生の輪郭（薄い赤）と フーリエ近似線（太いシアン）
            cv2.polylines(display_frame, [pts_raw], True, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.polylines(display_frame, [pts_smooth], True, (255, 255, 0), 3, lineType=cv2.LINE_AA)
            
            cv2.putText(display_frame, f"Fourier Freqs: {self.num_frequencies}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, "Press 'UP/DOWN' to change freqs, 'd' to toggle background", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 画面への表示
            cv2.imshow("Fourier Face Camera", frame)

            # 1ミリ秒キー入力を待ち、'q'が押されたらループを抜ける
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'): # 背景切り替え
                drawing_mode = not drawing_mode
            elif key == 0: # 上矢印キー（OpenCVの環境によっては 82）
                self.num_frequencies = min(self.num_frequencies + 2, len(self.contour_indices))
            elif key == 1: # 下矢印キー（OpenCVの環境によっては 84）
                self.num_frequencies = max(self.num_frequencies - 2, 2)

        self.cleanup()

    def cleanup(self) -> None:
        """
        リソースの解放
        """
        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed.")

if __name__ == "__main__":
    app = FourierFaceCamera()
    app.run()