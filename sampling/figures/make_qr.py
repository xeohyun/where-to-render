"""
make_qr.py — 판넬의 웹 데모 QR 생성 (2026-11-12)
================================================
웹 데모를 Claude 아티팩트(로그인 필요)에서 GitHub Pages 공개 주소로 옮기면서 QR을 다시 만든다.
env: python-pptx venv(qrcode 포함)
    <venv>/bin/python sampling/figures/make_qr.py [--url ...]
"""
import argparse
from pathlib import Path

import qrcode

HERE = Path(__file__).resolve().parent
URL = "https://xeohyun.github.io/where-to-render/"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(HERE / "poster_qr_demo.png"))
    a = ap.parse_args()
    q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=2)
    q.add_data(a.url)
    q.make(fit=True)
    q.make_image(fill_color="black", back_color="white").save(a.out)
    print("saved", a.out, a.url)
