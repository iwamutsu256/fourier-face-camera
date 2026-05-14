import cv2
import numpy as np

class FourierFaceCamera:
    """
    リアルタイム映像を取得し、フーリエ変換によるエフェクトをかけて仮想カメラに出力するクラス
    """
    def __init__(self, camera_id: int = 0):
        # cv2.VideoCaptureはOSのカメラAPIを叩き、デバイスと接続するクラス
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ (ID: {camera_id}) にアクセスできません。")
            
        # カメラの解像度を取得（デバッグ用）
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera initialized. Resolution: {width}x{height}")

    def run(self) -> None:
        """
        メイン処理ループ
        """
        print("[INFO] Starting main loop. Press 'q' to exit.")
        while True:
            # ret: フレームの取得に成功したか (bool)
            # frame: 取得した画像データ (np.ndarray)
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # ---------------------------------------------------------
            # [学習用デバッグ表示] データ構造の確認（理解できたらコメントアウト可）
            # ---------------------------------------------------------
            # shape: (Height, Width, Channels)  ex: (720, 1280, 3)
            # dtype: uint8 (0~255の8ビット符号なし整数)
            cv2.putText(
                frame, 
                f"Shape: {frame.shape} Dtype: {frame.dtype}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), # BGRなので緑色
                2
            )

            # 画面への表示
            cv2.imshow("Fourier Face Camera", frame)

            # 1ミリ秒キー入力を待ち、'q'が押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self) -> None:
        """
        リソースの解放
        """
        self.cap.release()
        cv2.imshow("Fourier Face Camera", np.zeros((10,10,3), dtype=np.uint8)) # Macでのウィンドウ残像バグ対策
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed.")

if __name__ == "__main__":
    app = FourierFaceCamera()
    app.run()