"""セグメンテーションモデル準備処理を確認するテスト。"""

from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from segmentation.model import prepare_model


class SegmentationModelTest(unittest.TestCase):
    """モデルファイル準備処理を確認する。"""

    def test_returns_existing_model_path_without_download(self):
        """
        モデルファイルが存在する場合にダウンロードしないことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            一時ファイルを作成する。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.tflite"
            model_path.write_bytes(b"dummy")

            def fail_downloader(model_url: str, output_path: str) -> None:
                raise AssertionError("既存モデルではダウンロードしない")

            result = prepare_model(str(model_path), "https://example.com/model.tflite", fail_downloader)

            self.assertEqual(str(model_path), result)

    def test_downloads_model_when_file_is_missing(self):
        """
        モデルファイルが存在しない場合にダウンロード関数を呼ぶことを確認する。

        Args:
            なし

        Returns:
            なし

        Side Effects:
            一時ファイルを作成する。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.tflite"
            calls = []

            def fake_downloader(model_url: str, output_path: str) -> None:
                calls.append((model_url, output_path))
                Path(output_path).write_bytes(b"downloaded")

            result = prepare_model(str(model_path), "https://example.com/model.tflite", fake_downloader)

            self.assertEqual(str(model_path), result)
            self.assertEqual([("https://example.com/model.tflite", str(model_path))], calls)
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
