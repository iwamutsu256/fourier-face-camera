"""Fourier Face Camera の設定値を定義する。"""

# カメラ設定
DEFAULT_CAMERA_ID = 0
FALLBACK_FPS = 30

# モデル設定
SELFIE_SEGMENTER_MODEL_PATH = "selfie_segmenter.tflite"
SELFIE_SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

# 線画設定
DEFAULT_LINE_COLOR_RGB = (0, 255, 255)
DEFAULT_SKETCH_THRESHOLD = 30
DEFAULT_NUM_FREQUENCIES = 10
MIN_CONTOUR_POINTS = 4

# マスク処理設定
ERODE_KERNEL_SIZE = (5, 5)
ERODE_ITERATIONS = 2

# スケッチ抽出設定
GAUSSIAN_BLUR_KERNEL_SIZE = (21, 21)
SKETCH_DIVIDE_SCALE = 256
SKETCH_MAX_VALUE = 255
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200
CLOSE_KERNEL_SIZE = (3, 3)
MAX_INNER_CONTOURS = 25

# 描画設定
SILHOUETTE_LINE_THICKNESS = 3
INNER_LINE_THICKNESS = 2

# モニター表示設定
MONITOR_WINDOW_NAME = "Fourier Face Camera (Monitor)"
EXIT_KEY = "q"

# 実行時の初期設定
APP_LINE_COLOR_RGB = (255, 0, 255)
APP_SKETCH_THRESHOLD = 60
