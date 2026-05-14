# 🎭 Real-time Fourier Edge Camera

リアルタイムのWebカメラ映像から**「人物だけをAIで切り抜き、眼鏡や服のシワを含むその人自身の見た目のエッジ」**を抽出し、**離散フーリエ変換 (FFT)** を用いて一筆書きのような滑らかなジェネラティブアート線画に変換します。
さらに、生成されたアート映像を仮想カメラとして出力し、ZoomやGoogle MeetなどのWeb会議で利用可能にするインタラクティブ・エンジニアリング・プロジェクトです。

![Demo GIF](https://github.com/user-attachments/assets/f0176be2-33f7-47f1-964b-88af3e16a97c)

## ✨ Core Features

- **AI Selfie Segmentation:** 部屋の背景（時計やポスターなど）をAIで完全に排除。人物と身につけているアイテムだけをターゲットにします。
- **Dual-Pass Edge Detection:** シルエット（外枠）の太い線と、顔のパーツ（内側）のディテール線を分けて抽出し、完璧な境界線を作り出します。
- **Fourier Smoothing (FFT):** 抽出したガタガタのピクセルエッジを複素平面にマッピングし、高周波ノイズをカットして逆変換 (IFFT) することで、数学的に滑らかな曲線へと昇華させます。
- **Virtual Camera Integration:** 生成されたアート線画の映像をOSレベルの仮想カメラとしてストリーミングします。

---

## 🚀 Getting Started (導入方法)

### Prerequisites (前提条件)

1. **Python 3.10+**
2. **[OBS Studio](https://obsproject.com/ja)**: 映像をWeb会議ツールに流し込むための「仮想カメラドライバ」として使用します。インストールするだけで準備完了です。

### Installation (インストール手順)

1. リポジトリをクローンします。

```bash
git clone https://github.com/iwamutsu256/fourier-face-camera.git
cd fourier-face-camera
```

2. 仮想環境を作成し、アクティベートします。

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. 必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use (使用方法)

### 1. プログラムの実行

ターミナルで以下のコマンドを実行します。
_(※初回起動時は、Googleのサーバーから約数MBのAIモデルファイルが自動ダウンロードされます。)_

```bash
python src/main.py
```

### 2. Web会議での利用 (Zoom / Google Meet)

1. プログラムを実行したまま、ZoomやGoogle Meetを開きます。
2. カメラの設定（ビデオ設定）を開き、デバイス一覧から **`OBS Virtual Camera`** を選択します。
3. あなた自身が滑らかなフーリエアートの線画となって会議に映し出されます！

### 3. 操作・終了方法

- **手元モニターの終了**: モニターウィンドウ（Fourier Face Camera）を選択した状態で、キーボードの **`q`** キーを押すと安全に終了します。
- **線の色の変更**: `src/main.py` の一番下にある `line_color_rgb=(0, 255, 255)` のRGB値を変更することで、好きな色にカスタマイズできます。

---

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV**: エッジ検出 (Canny法)、ガウシアンブラー、モルフォロジー演算 (Closing/Erosion)、レンダリング
- **MediaPipe (Tasks API)**: 軽量かつ高速な人物切り抜き (`ImageSegmenter`)
- **NumPy**: 行列操作、高速フーリエ変換 (`np.fft`)
- **pyvirtualcam**: 仮想カメラストリーミング

---

## 📝 Learning Logs

このリポジトリは、ただのソースコード置き場ではなく、「なぜその処理が必要なのか」「内部で何が起きているのか」を深く理解しながら進めた学習ログとして構成されています。初期のアプローチ（FaceMesh）から現在のハイブリッド構成に至るまでの技術的な試行錯誤が記録されています。

- [Step 1: OpenCVとNumPyによる映像取得](./docs/step01_opencv.md)
- [Step 2: MediaPipe FaceMeshによる輪郭抽出](./docs/step02_facemesh.md)
- [Step 3: 高速フーリエ変換 (FFT) による線画近似](./docs/step03_fft.md)
- [Step 3.5: AIセグメンテーションとエッジ検出のハイブリッド化](./docs/step03.5_segmentation_fft.md)
- [Step 4: 仮想カメラへの出力と本番化](./docs/step04_virtualcam.md)
