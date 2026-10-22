#!/usr/bin/env python3
"""Download a PaddleOCR DB text detector (Apache-2.0) as ONNX and write the
manifest reco-anon requires (kind "text-db").

    python scripts/fetch_ppocr.py --version v5 --out models/ppocr-det-v5.onnx

Source: https://huggingface.co/monkt/paddleocr-onnx (Apache-2.0 model card,
ONNX conversions of PaddleOCR PP-OCRv3/v5 detection models).
Check the printed sha256 against the repository before relying on it.
"""
import argparse
import hashlib
import json
import os
import urllib.request

BASE = "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v3", "v5"], default="v5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = f"{BASE}/{args.version}/det.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    manifest = {
        "id": f"ppocr-det-{args.version}",
        "sha256": sha,
        "license": "Apache-2.0 (PaddleOCR; ONNX conversion monkt/paddleocr-onnx)",
        "kind": "text-db",
        "source": url,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} and {args.out}.json (sha256 {sha})")


if __name__ == "__main__":
    main()
