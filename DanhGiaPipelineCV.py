"""Đánh giá pipeline: YOLO một lớp → HOG/HSV → SVM → kiểm tra HSV/Canny."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from joblib import load
from ultralytics import YOLO

from cv_pipeline import accept_yolo_prediction, enhance_contrast, feature_vector, verify_sign_region

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "vnts_detector" / "weights" / "best_detector.pt"
SVM_PATH = ROOT / "svm_model.joblib"


def iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - inter
    return inter / union if union else 0.0


def ground_truth(label_path, width, height):
    targets = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, x, y, w, h = map(float, line.split())
        targets.append((int(class_id), ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)))
    return targets


def score(predictions, targets):
    matched, tp, fp = set(), 0, 0
    for class_id, box in predictions:
        candidates = [(iou(box, target_box), index) for index, (target_class, target_box) in enumerate(targets) if target_class == class_id and index not in matched]
        best_iou, index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched.add(index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(targets) - tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not MODEL_PATH.exists() or not SVM_PATH.exists():
        raise FileNotFoundError("Cần best_detector.pt và svm_model.joblib trước khi đánh giá.")

    detector = YOLO(str(MODEL_PATH))
    classifier = load(SVM_PATH)
    split_file = ROOT / "data_detector" / "splits" / "test.txt"
    paths = [ROOT / "data" / "images" / Path(line).name for line in split_file.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        paths = paths[:args.limit]

    totals = [0, 0, 0]
    for image_path in paths:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        enhanced = enhance_contrast(image)
        result = detector.predict(enhanced, imgsz=640, conf=0.25, verbose=False)[0]
        predictions = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = enhanced[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size and accept_yolo_prediction(float(box.conf[0]), verify_sign_region(crop)):
                class_id = int(classifier.predict(feature_vector(crop).reshape(1, -1))[0])
                predictions.append((class_id, (x1, y1, x2, y2)))
        values = score(predictions, ground_truth(ROOT / "data" / "labels" / f"{image_path.stem}.txt", width, height))
        totals = [total + value for total, value in zip(totals, values)]

    tp, fp, fn = totals
    print("Pipeline YOLO một lớp → HOG/HSV → SVM → HSV/Canny")
    print(f"Precision={tp / max(tp + fp, 1):.3f}, Recall={tp / max(tp + fn, 1):.3f}, TP={tp}, FP={fp}, FN={fn}")


if __name__ == "__main__":
    main()
