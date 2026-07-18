"""Huấn luyện bộ phân loại biển báo Việt Nam từ nhãn bounding-box của VNTS.

Ví dụ: python HuanLuyenMoHinh.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv_pipeline import enhance_contrast, feature_vector

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def normalized_to_box(values: list[str], width: int, height: int) -> tuple[int, int, int, int]:
    _, x_center, y_center, box_width, box_height = map(float, values[:5])
    left = max(0, int((x_center - box_width / 2) * width))
    top = max(0, int((y_center - box_height / 2) * height))
    right = min(width, int((x_center + box_width / 2) * width))
    bottom = min(height, int((y_center + box_height / 2) * height))
    return left, top, right, bottom


def load_samples(data_dir: Path) -> list[tuple[Path, int, tuple[float, float, float, float]]]:
    image_dir, label_dir = data_dir / "images", data_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Cần có data/images và data/labels của bộ VNTS.")
    samples = []
    for annotation_path in sorted(label_dir.glob("*.txt")):
        image_path = next((candidate for candidate in (
            image_dir / f"{annotation_path.stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
        ) if candidate.exists()), None)
        if image_path is None:
            continue
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 5:
                samples.append((image_path, int(values[0]), tuple(map(float, values[1:5]))))
    if not samples:
        raise RuntimeError("Không tìm thấy annotation bounding-box hợp lệ.")
    return samples


def extract_features(samples: list[tuple[Path, int, tuple[float, float, float, float]]]) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    cache: dict[Path, np.ndarray] = {}
    for index, (image_path, label, coordinates) in enumerate(samples, 1):
        image = cache.get(image_path)
        if image is None:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            cache[image_path] = image
        height, width = image.shape[:2]
        left, top, right, bottom = normalized_to_box([str(label), *map(str, coordinates)], width, height)
        crop = image[top:bottom, left:right]
        if crop.size:
            features.append(feature_vector(enhance_contrast(crop)))
            targets.append(label)
        if index % 1000 == 0:
            print(f"Đã xử lý {index}/{len(samples)} biển báo")
    return np.asarray(features), np.asarray(targets)


def save_reports(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], class_names: list[str], reports: Path) -> float:
    reports.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    (reports / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Ma trận nhầm lẫn VNTS — accuracy={accuracy:.3f}")
    plt.xlabel("Lớp dự đoán")
    plt.ylabel("Lớp thật")
    plt.tight_layout()
    plt.savefig(reports / "confusion_matrix.png", dpi=180)
    plt.close()
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số bounding box để chạy thử nhanh")
    parser.add_argument("--c-values", type=float, nargs="+", default=[1, 10, 30])
    args = parser.parse_args()
    samples = load_samples(args.data_dir)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Tìm thấy {len(samples)} bounding box, {len(Counter(item[1] for item in samples))} lớp.")
    features, targets = extract_features(samples)
    labels = sorted(set(targets))
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )
    reports = ROOT / "reports"
    sweep, best_model, best_accuracy = [], None, -1.0
    for c_value in args.c_values:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c_value, gamma="scale"))
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        sweep.append({"C": c_value, "accuracy": accuracy})
        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy
    reports.mkdir(exist_ok=True)
    pd.DataFrame(sweep).to_csv(reports / "svm_parameter_sweep.csv", index=False)
    names = (args.data_dir / "classes_vie.txt").read_text(encoding="utf-8").splitlines()
    predictions = best_model.predict(x_test)
    accuracy = save_reports(y_test, predictions, labels, [names[label] for label in labels], reports)
    dump(best_model, ROOT / "svm_model.joblib")
    print(f"Đã lưu model: {ROOT / 'svm_model.joblib'}")
    print(f"C tốt nhất: {max(sweep, key=lambda item: item['accuracy'])['C']}; accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
