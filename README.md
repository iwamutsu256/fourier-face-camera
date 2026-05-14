# 🎭 Real-time Fourier Edge Camera

リアルタイムのWebカメラ映像から**「人物だけをAIで切り抜き、眼鏡や服のシワを含むその人自身の見た目のエッジ」**を抽出し、**離散フーリエ変換 (DFT)** を用いて一筆書きのような滑らかなジェネラティブアート線画に変換。さらにそれを仮想カメラとして出力し、ZoomなどのWeb会議で利用可能にするインタラクティブ・エンジニアリング・プロジェクト。

## ✨ Core Features

1. **AI Selfie Segmentation:** 部屋の背景（時計やポスターなど）をAIで完全に排除。人物と身につけているアイテムだけをターゲットにします。
2. **Dual-Pass Edge Detection:** シルエット（外枠）の太い線と、顔のパーツ（内側）のディテール線を分けて抽出し、完璧な境界線を作り出します。
3. **Fourier Smoothing (FFT):** 抽出したガタガタのピクセルエッジを複素平面にマッピングし、高周波ノイズをカットして逆変換 (IFFT) することで、数学的に滑らかな曲線へと昇華させます。
4. **Virtual Camera Integration:** 生成された黒背景×アート線画の映像をOSレベルの仮想カメラとしてストリーミングします。

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV**: エッジ検出 (Canny法)、ガウシアンブラー、モルフォロジー演算 (Closing/Erosion)、レンダリング
- **MediaPipe (Tasks API)**: 軽量かつ高速な人物切り抜き (`ImageSegmenter`)
- **NumPy**: 行列操作、高速フーリエ変換 (`np.fft`)
- **pyvirtualcam**: 仮想カメラストリーミング

## 📝 Learning Logs

このリポジトリは、ただのソースコード置き場ではなく、私が「なぜその処理が必要なのか」「内部で何が起きているのか」を深く理解しながら進めた学習ログです。初期のアプローチ（FaceMesh）から現在のハイブリッド構成に至るまでの試行錯誤が記録されています。

- [Step 1: OpenCVとNumPyによる映像取得](./docs/step01_opencv.md)
- [Step 2: MediaPipe FaceMeshによる輪郭抽出](./docs/step02_facemesh.md) (※初期アプローチ)
- [Step 3: 高速フーリエ変換 (FFT) による線画近似](./docs/step03_fft.md)
- [Step 3.5: AIセグメンテーションとエッジ検出のハイブリッド化](./docs/step03.5_segmentation_fft.md)
- [Step 4: 仮想カメラへの出力と本番化](./docs/step04_virtualcam.md)
