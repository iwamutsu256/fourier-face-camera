"""Fourier Face Camera の起動エントリーポイント。"""

from app import create_default_app


def main() -> None:
    """
    アプリケーションを起動する。

    Args:
        なし

    Returns:
        なし

    Side Effects:
        カメラ入力、線画化、仮想カメラ出力を開始する。
    """
    app = create_default_app()
    app.run()


if __name__ == "__main__":
    main()
