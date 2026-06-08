"""セグメンテーションモデルの準備処理を定義する。"""

import os
import urllib.request
from collections.abc import Callable

from config import settings


def prepare_model(
    model_path: str = settings.SELFIE_SEGMENTER_MODEL_PATH,
    model_url: str = settings.SELFIE_SEGMENTER_MODEL_URL,
    downloader: Callable[[str, str], object] = urllib.request.urlretrieve,
) -> str:
    """
    セグメンテーションモデルを準備する。

    Args:
        model_path:
            モデルファイルの保存先
        model_url:
            モデルファイルの取得元URL
        downloader:
            モデルをダウンロードする関数

    Returns:
        モデルファイルのパス

    Side Effects:
        モデルファイルが存在しない場合はダウンロードする。
    """
    if os.path.exists(model_path):
        return model_path

    print("[INFO] Downloading Selfie Segmenter model. Please wait...")
    downloader(model_url, model_path)
    print("[INFO] Model download complete.")
    return model_path
