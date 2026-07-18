"""Nhãn biển báo Việt Nam lấy trực tiếp từ bộ dữ liệu VNTS."""

from pathlib import Path

_labels_file = Path(__file__).resolve().parent / "data" / "classes_vie.txt"

if _labels_file.exists():
    class_names = {
        index: name.strip()
        for index, name in enumerate(_labels_file.read_text(encoding="utf-8").splitlines())
        if name.strip()
    }
else:
    class_names = {}
