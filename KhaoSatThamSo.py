"""Tạo ảnh khảo sát tham số cho báo cáo/lab report.

Ví dụ: python KhaoSatThamSo.py --image data/example.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from cv_pipeline import colour_mask, enhance_contrast

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="Ảnh thực tế chứa biển báo")
    parser.add_argument("--output", type=Path, default=Path("reports/parameter_sweep.png"))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Không đọc được {args.image}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(12, 11))
    for column, clip in enumerate((1.0, 2.0, 4.0)):
        enhanced = enhance_contrast(image, clip)
        axes[0, column].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[0, column].set_title(f"CLAHE clip={clip}")
    for column, saturation in enumerate((50, 70, 100)):
        axes[1, column].imshow(colour_mask(image, saturation), cmap="gray")
        axes[1, column].set_title(f"HSV S_min={saturation}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for column, low_threshold in enumerate((30, 60, 100)):
        axes[2, column].imshow(cv2.Canny(gray, low_threshold, low_threshold * 2.5), cmap="gray")
        axes[2, column].set_title(f"Canny low={low_threshold}")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle("Khảo sát tham số: mỗi kỹ thuật dùng 3 giá trị", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
