"""Huấn luyện YOLO trên bộ biển báo Việt Nam VNTS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu hoặc 0 nếu máy có GPU CUDA")
    parser.add_argument("--data", type=Path, default=ROOT / "vnts_detector.yaml")
    parser.add_argument("--name", default="vnts_detector")
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình dữ liệu: {args.data}")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=str(ROOT / "runs"),
        name=args.name, patience=20, pretrained=True, plots=True,
    )


if __name__ == "__main__":
    main()
