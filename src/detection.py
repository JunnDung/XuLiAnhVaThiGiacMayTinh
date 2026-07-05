
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============== BASIC UTILITIES ==============

def load_image(path: str | os.PathLike) -> np.ndarray:
    """Read an image in BGR format. OpenCV reads BGR, not RGB."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return img


def save_image(img: np.ndarray, out_path: str | os.PathLike) -> None:
    """Save an image and create the parent folder when needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"Khong luu duoc anh: {out_path}")
    encoded.tofile(str(out_path))


def list_image_files(folder: str | os.PathLike) -> list[Path]:
    """Return image files under a folder, sorted for reproducible output."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _odd_kernel_size(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def to_gray_blur(img_bgr: np.ndarray, blur_ksize: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Convert BGR image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = _odd_kernel_size(blur_ksize)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    return gray, blur


# ============== OPTION A: CANNY + HOUGH ==============

def canny_edge(img_gray: np.ndarray, low: int, high: int) -> np.ndarray:
    """Detect edges using Canny hysteresis thresholds."""
    return cv2.Canny(img_gray, int(low), int(high))


def hough_circles(img_gray: np.ndarray,
                  dp: float = 1.2,
                  min_dist: int = 50,
                  param1: int = 150,
                  param2: int = 30,
                  min_radius: int = 20,
                  max_radius: int = 100) -> np.ndarray:
    """
    Detect circles by Hough Transform.

    Returns an array of shape (N, 3), each row is (cx, cy, r).
    """
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=int(min_dist),
        param1=int(param1),
        param2=int(param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return np.zeros((0, 3), dtype=np.float32)
    return circles[0].astype(np.float32)


def _circle_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate overlap for circle candidates by their bounding boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2], a[1] - a[2], a[0] + a[2], a[1] + a[2]
    bx1, by1, bx2, by2 = b[0] - b[2], b[1] - b[2], b[0] + b[2], b[1] + b[2]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def filter_circles(circles: np.ndarray,
                   max_circles: int | None = 20,
                   overlap_threshold: float = 0.45) -> np.ndarray:
    """
    Remove heavily overlapping circle candidates and optionally keep top-N.

    OpenCV does not return a confidence score, so larger circles are kept first.
    """
    if len(circles) == 0:
        return circles.astype(np.float32)

    ordered = sorted(circles.astype(np.float32), key=lambda c: c[2], reverse=True)
    kept: list[np.ndarray] = []
    for circle in ordered:
        if all(_circle_iou(circle, prev) < overlap_threshold for prev in kept):
            kept.append(circle)
        if max_circles is not None and len(kept) >= max_circles:
            break
    return np.array(kept, dtype=np.float32)


def circle_bbox(img: np.ndarray, cx: int, cy: int, r: int, pad: float = 0.15) -> tuple[int, int, int, int]:
    """Return clipped bounding box (x1, y1, x2, y2) around a circle."""
    h, w = img.shape[:2]
    r_pad = max(1, int(round(r * (1 + pad))))
    x1 = max(0, int(cx) - r_pad)
    y1 = max(0, int(cy) - r_pad)
    x2 = min(w, int(cx) + r_pad)
    y2 = min(h, int(cy) + r_pad)
    return x1, y1, x2, y2


def crop_by_circle(img: np.ndarray, cx: int, cy: int, r: int, pad: float = 0.15) -> np.ndarray:
    """Crop the region around a circle with padding."""
    x1, y1, x2, y2 = circle_bbox(img, cx, cy, r, pad)
    return img[y1:y2, x1:x2]


def draw_circles_on_image(img_bgr: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """Draw detected circles on a copy of the input image."""
    img_out = img_bgr.copy()
    for cx, cy, r in np.uint16(np.around(circles)):
        cv2.circle(img_out, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(img_out, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    return img_out


def hough_lines(img_gray: np.ndarray,
                rho: float = 1,
                theta: float = np.pi / 180,
                threshold: int = 50,
                min_line_length: int = 50,
                max_line_gap: int = 10) -> np.ndarray:
    """Detect line segments with probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(
        img_gray,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.zeros((0, 1, 4), dtype=np.int32)
    return lines


def detect_circles_pipeline(img_bgr: np.ndarray,
                            canny_low: int = 50,
                            canny_high: int = 150,
                            hough_param2: int = 30,
                            blur_ksize: int = 5,
                            dp: float = 1.2,
                            min_dist: int = 50,
                            min_radius: int = 20,
                            max_radius: int = 100,
                            max_circles: int | None = 20) -> dict:
    """Pipeline A: BGR -> Gray -> Blur -> Canny -> Hough Circle -> Crop."""
    gray, blur = to_gray_blur(img_bgr, blur_ksize)
    edges = canny_edge(blur, canny_low, canny_high)
    raw_circles = hough_circles(
        blur,
        dp=dp,
        min_dist=min_dist,
        param1=canny_high,
        param2=hough_param2,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    circles = filter_circles(raw_circles, max_circles=max_circles)

    crops = []
    bboxes = []
    for cx, cy, r in circles:
        x1, y1, x2, y2 = circle_bbox(img_bgr, int(cx), int(cy), int(r))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            bboxes.append((x1, y1, x2, y2))

    return {
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "circles_raw": raw_circles,
        "circles": circles,
        "bboxes": bboxes,
        "crops": crops,
        "num_signs": len(crops),
        "image_with_circles": draw_circles_on_image(img_bgr, circles),
    }


# ============== OPTION B: HSV MASK + MARKER-CONTROLLED WATERSHED ==============

def traffic_sign_color_mask(img_bgr: np.ndarray,
                            include_red: bool = True,
                            include_blue: bool = True,
                            include_yellow: bool = True) -> np.ndarray:
    """Create a combined HSV mask for common traffic sign colors."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []

    if include_red:
        red1 = cv2.inRange(hsv, np.array((0, 70, 50), dtype=np.uint8), np.array((10, 255, 255), dtype=np.uint8))
        red2 = cv2.inRange(hsv, np.array((170, 70, 50), dtype=np.uint8), np.array((180, 255, 255), dtype=np.uint8))
        masks.append(cv2.bitwise_or(red1, red2))
    if include_blue:
        masks.append(cv2.inRange(hsv, np.array((95, 60, 50), dtype=np.uint8), np.array((130, 255, 255), dtype=np.uint8)))
    if include_yellow:
        masks.append(cv2.inRange(hsv, np.array((15, 60, 80), dtype=np.uint8), np.array((40, 255, 255), dtype=np.uint8)))

    if not masks:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    mask = masks[0]
    for next_mask in masks[1:]:
        mask = cv2.bitwise_or(mask, next_mask)
    return mask


def clean_binary_mask(mask: np.ndarray, opening_ksize: int = 5) -> np.ndarray:
    """Clean a binary mask using opening and closing."""
    k = _odd_kernel_size(opening_ksize)
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean


def make_marker_watershed(img_bgr: np.ndarray,
                          hsv_lower: tuple | None = None,
                          hsv_upper: tuple | None = None,
                          opening_ksize: int = 5) -> np.ndarray:
    """
    Segment likely traffic-sign regions with marker-controlled Watershed.

    If hsv_lower/hsv_upper are provided, only that HSV range is used.
    Otherwise, red/blue/yellow traffic-sign colors are combined.
    Returns a binary foreground mask where 255 means detected foreground.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if hsv_lower is not None and hsv_upper is not None:
        base_mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    else:
        base_mask = traffic_sign_color_mask(img_bgr)

    mask_clean = clean_binary_mask(base_mask, opening_ksize)
    if cv2.countNonZero(mask_clean) == 0:
        return mask_clean

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_clean, kernel, iterations=3)
    dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr.copy(), markers)

    watershed_mask = np.zeros(mask_clean.shape, dtype=np.uint8)
    watershed_mask[markers > 1] = 255
    watershed_mask = clean_binary_mask(watershed_mask, 3)
    return watershed_mask


def crop_by_mask(img_bgr: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop the largest connected foreground region in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return img_bgr[y1:y2, x1:x2]


def detect_watershed_pipeline(img_bgr: np.ndarray,
                              hsv_lower: tuple | None = None,
                              hsv_upper: tuple | None = None,
                              opening_ksize: int = 5,
                              min_area: int = 100,
                              pad: int = 8) -> dict:
    """Pipeline B: HSV color mask -> morphology -> marker-controlled Watershed -> Crop."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if hsv_lower is not None and hsv_upper is not None:
        mask_color = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    else:
        mask_color = traffic_sign_color_mask(img_bgr)
    mask_clean = clean_binary_mask(mask_color, opening_ksize)
    watershed_mask = make_marker_watershed(img_bgr, hsv_lower, hsv_upper, opening_ksize)

    contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    bboxes = []
    h_img, w_img = img_bgr.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            bboxes.append((x1, y1, x2, y2))

    img_contours = img_bgr.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    return {
        "hsv": hsv,
        "mask_color": mask_color,
        "mask_clean": mask_clean,
        "mask": watershed_mask,
        "contours": contours,
        "bboxes": bboxes,
        "crops": crops,
        "num_signs": len(crops),
        "img_contours": img_contours,
    }


# ============== PARAMETER SURVEYS ==============

def survey_canny_thresholds(img_bgr: np.ndarray,
                            thresholds: Iterable[tuple[int, int]] | None = None,
                            hough_param2: int = 30) -> dict:
    """Survey at least three Canny threshold pairs."""
    if thresholds is None:
        thresholds = [(30, 80), (50, 150), (80, 200)]

    _, blur = to_gray_blur(img_bgr)
    results = {}
    for low, high in thresholds:
        edges = canny_edge(blur, low, high)
        circles = hough_circles(blur, param1=high, param2=hough_param2)
        circles = filter_circles(circles, max_circles=20)
        results[(low, high)] = {
            "edges": edges,
            "edge_pixels": int(cv2.countNonZero(edges)),
            "num_circles": len(circles),
            "circles": circles,
            "image_with_circles": draw_circles_on_image(img_bgr, circles),
        }
    return results


def survey_hough_param2(img_bgr: np.ndarray,
                        param2_values: Iterable[int] | None = None,
                        canny_high: int = 150) -> dict:
    """Survey at least three Hough accumulator thresholds."""
    if param2_values is None:
        param2_values = [15, 30, 50]

    _, blur = to_gray_blur(img_bgr)
    results = {}
    for p2 in param2_values:
        circles = hough_circles(blur, param1=canny_high, param2=p2)
        circles = filter_circles(circles, max_circles=20)
        results[int(p2)] = {
            "num_circles": len(circles),
            "circles": circles,
            "image_with_circles": draw_circles_on_image(img_bgr, circles),
        }
    return results


def survey_watershed_kernel(img_bgr: np.ndarray,
                            kernel_sizes: Iterable[int] | None = None) -> dict:
    """Survey morphology kernel sizes for the watershed branch."""
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 7]

    results = {}
    for k in kernel_sizes:
        out = detect_watershed_pipeline(img_bgr, opening_ksize=int(k))
        results[int(k)] = {
            "mask": out["mask"],
            "mask_pixels": int(cv2.countNonZero(out["mask"])),
            "num_regions": out["num_signs"],
            "img_contours": out["img_contours"],
        }
    return results


# ============== VISUALIZATION AND BATCH OUTPUT ==============

def show_images_grid(images_dict: dict, figsize: tuple = (15, 10), save_path: str | os.PathLike | None = None) -> None:
    """Display one or more images in a grid and optionally save the figure."""
    n = len(images_dict)
    rows = (n + 2) // 3
    cols = min(n, 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, images_dict.items()):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_detection_outputs(input_dir: str | os.PathLike = "data/processed",
                           crop_dir: str | os.PathLike = "data/cropped",
                           output_dir: str | os.PathLike = "outputs/detection",
                           fallback_raw_dir: str | os.PathLike = "data/raw") -> list[dict]:
    """
    Run Coder 2 pipeline on a folder and save overlays, masks, crops, and metadata.

    If data/processed is empty, data/raw is used so the demo remains runnable.
    """
    input_files = list_image_files(input_dir)
    source_dir = Path(input_dir)
    if not input_files:
        input_files = list_image_files(fallback_raw_dir)
        source_dir = Path(fallback_raw_dir)

    crop_dir = Path(crop_dir)
    output_dir = Path(output_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for image_path in input_files:
        img = load_image(image_path)
        stem = image_path.stem

        circle_out = detect_circles_pipeline(img)
        watershed_out = detect_watershed_pipeline(img)

        save_image(circle_out["image_with_circles"], output_dir / f"{stem}_hough_overlay.jpg")
        save_image(circle_out["edges"], output_dir / f"{stem}_canny_edges.png")
        save_image(watershed_out["mask"], output_dir / f"{stem}_watershed_mask.png")
        save_image(watershed_out["img_contours"], output_dir / f"{stem}_watershed_contours.jpg")

        # Hough crops are preferred for circular signs. Watershed crops are still exported.
        for method, out in (("hough", circle_out), ("watershed", watershed_out)):
            for idx, crop in enumerate(out["crops"], start=1):
                crop_name = f"{stem}_{method}_{idx:03d}.jpg"
                crop_path = crop_dir / "unknown" / crop_name
                save_image(crop, crop_path)
                x1, y1, x2, y2 = out["bboxes"][idx - 1]
                records.append({
                    "crop_id": crop_name,
                    "original_image": str(image_path.relative_to(source_dir.parent)),
                    "method": method,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": "unknown",
                })

    metadata_path = crop_dir / "metadata.csv"
    if records:
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Coder 2 traffic sign detection and cropping")
    parser.add_argument("--input", default="data/processed", help="Input folder. Falls back to data/raw if empty.")
    parser.add_argument("--crop-dir", default="data/cropped", help="Folder for cropped sign images.")
    parser.add_argument("--output", default="outputs/detection", help="Folder for intermediate visualizations.")
    args = parser.parse_args()

    records = save_detection_outputs(args.input, args.crop_dir, args.output)
    print(f"Processed crops: {len(records)}")
    print(f"Crops: {args.crop_dir}")
    print(f"Visualizations: {args.output}")


if __name__ == "__main__":
    main()
