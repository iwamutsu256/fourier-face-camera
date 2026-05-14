# 🎭 Real-time Fourier Face Camera

「Webカメラ映像」から「顔の輪郭」を抽出し、**離散フーリエ変換 (DFT)** を用いて一筆書きのような滑らかな近似線画を生成。それを仮想カメラとしてリアルタイム出力する、インタラクティブ・エンジニアリング・プロジェクト。

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV**: 映像取得・行列操作・レンダリング
- **MediaPipe**: 顔の3Dランドマーク（FaceMesh）高速抽出
- **NumPy**: 高速な行列演算、高速フーリエ変換 (`np.fft`)
- **pyvirtualcam**: 生成した配列空間をOSレベルの仮想カメラデバイスへストリーミング

## 📝 Learning Logs

このリポジトリは、ただのソースコード置き場ではなく、各技術の背景理解をまとめた学習ログとして機能します。

- [Step 1: OpenCVとNumPyによる映像取得](./docs/step01_opencv.md)
- Step 2: MediaPipe FaceMeshによる輪郭抽出 (WIP)
- Step 3: フーリエ変換による線画近似 (WIP)
- Step 4: 仮想カメラへのストリーミング (WIP)
