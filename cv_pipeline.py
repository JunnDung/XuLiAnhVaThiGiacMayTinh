"""Các bước thị giác máy tính dùng chung cho huấn luyện và ứng dụng."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

IMAGE_SIZE = (48, 48)
COLOR_BINS = 8


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Tăng tương phản cục bộ CLAHE trên kênh sáng."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(light), a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def colour_mask(image: np.ndarray, saturation_min: int = 70) -> np.ndarray:
    """Phân đoạn màu đỏ, lam và vàng thường gặp trên biển báo Việt Nam."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, saturation_min, 45]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([165, saturation_min, 45]), np.array([180, 255, 255])),
    )
    blue = cv2.inRange(hsv, np.array([95, saturation_min, 40]), np.array([135, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, saturation_min, 55]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red, blue), yellow)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def verify_sign_region(image: np.ndarray) -> dict[str, float | np.ndarray]:
    """Xác minh một bounding box YOLO bằng đặc trưng màu và cạnh cổ điển.

    CLAHE được áp dụng trước khi tính mask HSV và biên Canny. Điểm CV này được
    dùng để lọc các dự đoán YOLO có confidence thấp, không chỉ để minh họa.
    """
    enhanced = enhance_contrast(image)
    mask = colour_mask(enhanced)
    edges = cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), 60, 160)
    pixels = max(image.shape[0] * image.shape[1], 1)
    colour_ratio = cv2.countNonZero(mask) / pixels
    edge_ratio = cv2.countNonZero(edges) / pixels
    # Biển báo thường có màu bão hòa và cấu trúc biên rõ. Cả hai được chuẩn hóa
    # thành điểm 0..1 để hỗ trợ quyết định với prediction confidence thấp.
    colour_score = min(colour_ratio / 0.10, 1.0)
    edge_score = min(edge_ratio / 0.12, 1.0)
    cv_score = 0.6 * colour_score + 0.4 * edge_score
    return {
        "enhanced": enhanced,
        "mask": mask,
        "edges": edges,
        "colour_ratio": colour_ratio,
        "edge_ratio": edge_ratio,
        "cv_score": cv_score,
    }


def accept_yolo_prediction(confidence: float, verification: dict[str, float | np.ndarray]) -> bool:
    """Giữ box YOLO mạnh; box trung bình phải được HSV/Canny xác minh."""
    if confidence >= 0.60:
        return True
    return confidence >= 0.25 and float(verification["cv_score"]) >= 0.35


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


def detect_signs(image: np.ndarray, min_area_ratio: float = 0.00008) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """Đề xuất biển báo theo màu *và* hình học, loại bỏ cờ/chữ/đồ vật nền."""
    mask = colour_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    scored_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        ratio = width / max(height, 1)
        if not (0.75 <= ratio <= 1.35) or min(width, height) < 12:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1.0)
        is_round = circularity >= 0.48 and solidity >= 0.68
        is_polygonal_sign = len(polygon) in (3, 4) and solidity >= 0.75
        if not (is_round or is_polygonal_sign):
            continue
        padding = max(2, int(0.06 * max(width, height)))
        left, top = max(0, x - padding), max(0, y - padding)
        right = min(image.shape[1], x + width + padding)
        bottom = min(image.shape[0], y + height + padding)
        box = (left, top, right - left, bottom - top)
        scored_boxes.append((area * max(circularity, 0.5), box))
    # Biển cấm/giới hạn tốc độ thường là vòng tròn viền đỏ. Hough tránh nhầm
    # cờ hoặc chữ đỏ thành biển báo khi contour bị đứt bởi phần trắng bên trong.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 45]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([165, 70, 45]), np.array([180, 255, 255])),
    )
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (7, 7), 1.5), cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=max(35, min(image.shape[:2]) // 8),
        param1=100, param2=28, minRadius=12, maxRadius=min(image.shape[:2]) // 3,
    )
    if circles is not None:
        for center_x, center_y, radius in np.round(circles[0]).astype(int):
            ring = np.zeros_like(red_mask)
            cv2.circle(ring, (center_x, center_y), radius, 255, thickness=max(3, radius // 4))
            red_ratio = cv2.countNonZero(cv2.bitwise_and(red_mask, ring)) / max(cv2.countNonZero(ring), 1)
            if red_ratio < 0.22:
                continue
            left, top = max(0, center_x - radius), max(0, center_y - radius)
            right, bottom = min(image.shape[1], center_x + radius), min(image.shape[0], center_y + radius)
            scored_boxes.append((radius * radius * (1.0 + red_ratio), (left, top, right - left, bottom - top)))

    boxes = []
    for _, candidate in sorted(scored_boxes, reverse=True):
        if all(_iou(candidate, selected) < 0.35 for selected in boxes):
            boxes.append(candidate)
    return boxes, mask, edges


def detect_sign(image: np.ndarray, min_area_ratio: float = 0.005) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Phân đoạn màu, phát hiện cạnh Canny và chọn vùng biển báo lớn nhất."""
    boxes, mask, edges = detect_signs(image, min_area_ratio)
    if not boxes:
        height, width = image.shape[:2]
        return image.copy(), (0, 0, width, height), mask, edges
    x, y, width, height = boxes[0]
    return image[y : y + height, x : x + width].copy(), (x, y, width, height), mask, edges


def feature_vector(image: np.ndarray) -> np.ndarray:
    """Đặc trưng HOG (hình dạng) kết hợp histogram HSV (màu)."""
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm="L2-Hys", transform_sqrt=True,
    )
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = [
        cv2.calcHist([hsv], [channel], None, [COLOR_BINS], limits).flatten()
        for channel, limits in enumerate(((0, 180), (0, 256), (0, 256)))
    ]
    colour_features = np.concatenate(histograms).astype(np.float32)
    colour_features /= max(colour_features.sum(), 1.0)
    return np.concatenate((hog_features, colour_features)).astype(np.float32)


def hog_visualization(image: np.ndarray) -> np.ndarray:
    """Tạo ảnh HOG trực quan của ROI để hiển thị trong pipeline."""
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, visual = hog(
        gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm="L2-Hys", transform_sqrt=True, visualize=True,
    )
    return cv2.normalize(visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_image(image: np.ndarray) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    """Chạy toàn bộ pipeline và trả về các ảnh trung gian phục vụ demo/báo cáo."""
    enhanced = enhance_contrast(image)
    roi, box, mask, edges = detect_sign(enhanced)
    annotated = enhanced.copy()
    x, y, width, height = box
    cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return {"enhanced": enhanced, "mask": mask, "edges": edges, "roi": roi, "box": box, "annotated": annotated}
