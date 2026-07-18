"""Tạo dữ liệu YOLO một lớp: chỉ phát hiện vị trí biển báo, không phân loại."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data"
TARGET = ROOT / "data_detector"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)  # không nhân đôi dung lượng khi cùng ổ đĩa
    except OSError:
        target.write_bytes(source.read_bytes())


def main() -> None:
    image_source, label_source = SOURCE / "images", SOURCE / "labels"
    image_target, label_target = TARGET / "images", TARGET / "labels"
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_source.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError("Không tìm thấy data/images.")
    for image_path in image_paths:
        source_label = label_source / f"{image_path.stem}.txt"
        # Ảnh tự thêm để demo (ví dụ TEST.jpg) không có nhãn thì không dùng để train.
        if not source_label.exists():
            # Xóa bản sao cũ trong data_detector nếu script từng chạy dở dang.
            stale_image = image_target / image_path.name
            stale_label = label_target / source_label.name
            stale_image.unlink(missing_ok=True)
            stale_label.unlink(missing_ok=True)
            print(f"Bỏ qua ảnh không có nhãn: {image_path.name}")
            continue
        link_or_copy(image_path, image_target / image_path.name)
        target_label = label_target / source_label.name
        lines = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 5:
                lines.append("0 " + " ".join(values[1:5]))
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    paths = [path.resolve() for path in sorted(image_target.glob("*"))]
    random.Random(42).shuffle(paths)
    train_end, validation_end = int(len(paths) * 0.7), int(len(paths) * 0.9)
    split_dir = TARGET / "splits"
    split_dir.mkdir(exist_ok=True)
    groups = {"train": paths[:train_end], "val": paths[train_end:validation_end], "test": paths[validation_end:]}
    for name, group in groups.items():
        (split_dir / f"{name}.txt").write_text("\n".join(map(str, group)), encoding="utf-8")
    (ROOT / "vnts_detector.yaml").write_text(
        "\n".join((
            f"path: {TARGET.resolve().as_posix()}",
            "train: splits/train.txt",
            "val: splits/val.txt",
            "test: splits/test.txt",
            "nc: 1",
            "names: ['traffic_sign']",
            "",
        )), encoding="utf-8",
    )
    print(f"Đã tạo detector một lớp: train={len(groups['train'])}, val={len(groups['val'])}, test={len(groups['test'])}")


if __name__ == "__main__":
    main()


