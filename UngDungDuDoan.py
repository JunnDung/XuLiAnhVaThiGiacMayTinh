"""Demo Streamlit nhận diện biển báo Việt Nam bằng YOLO đã fine-tune."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from joblib import load
from ultralytics import YOLO

from cv_pipeline import accept_yolo_prediction, enhance_contrast, feature_vector, hog_visualization, verify_sign_region
from labels import class_names

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "vnts_detector" / "weights" / "best_detector.pt"
SVM_PATH = ROOT / "svm_model.joblib"

st.set_page_config(page_title="Nhận diện biển báo Việt Nam", page_icon="🚦", layout="wide")


@st.cache_resource
def load_model(path: Path) -> YOLO:
    return YOLO(str(path))


@st.cache_resource
def load_classifier(path: Path):
    return load(path)


st.title("🚦 Nhận diện biển báo giao thông Việt Nam")
st.caption("CLAHE → YOLO định vị box → HOG + HSV histogram + SVM phân loại → HSV/Canny xác minh")

if not MODEL_PATH.exists() or not SVM_PATH.exists():
    st.error("Cần có cả `runs/vnts_detector/weights/best_detector.pt` (YOLO một lớp) và `svm_model.joblib`.")
    st.stop()
model = load_model(MODEL_PATH)
classifier = load_classifier(SVM_PATH)

confidence = st.slider("Ngưỡng tin cậy", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
use_clahe = st.checkbox("Áp dụng CLAHE trước YOLO", value=True)
uploaded_file = st.file_uploader("Tải ảnh đường phố (JPG, JPEG hoặc PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.caption("Model chỉ hiển thị khung khi độ tin cậy lớn hơn ngưỡng bạn chọn.")
    st.stop()

image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if image is None:
    st.error("Không thể đọc ảnh đã tải lên.")
    st.stop()

enhanced = enhance_contrast(image)
prediction_input = enhanced if use_clahe else image
prediction = model.predict(source=prediction_input, imgsz=640, conf=confidence, verbose=False)[0]
annotated = image.copy()
detected = []
rejected = []
accepted = []
for number, box in enumerate(prediction.boxes, 1):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    score = float(box.conf[0])
    crop = enhanced[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        continue
    class_id = int(classifier.predict(feature_vector(crop).reshape(1, -1))[0])
    name = class_names.get(class_id, f"Lớp {class_id}")
    verification = verify_sign_region(crop)
    cv_score = float(verification["cv_score"])
    if not accept_yolo_prediction(score, verification):
        rejected.append(f"{name} ({score:.0%}, điểm CV {cv_score:.0%})")
        continue
    number_final = len(detected) + 1
    detected.append(f"Biển {number_final}: {name} ({score:.0%}, kiểm tra CV {cv_score:.0%})")
    accepted.append({"number": number_final, "name": name, "score": score, "cv_score": cv_score, "crop": crop, "verification": verification})
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"#{number_final}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

first, second, third = st.columns(3)
first.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="1. Ảnh đầu vào", use_container_width=True)
second.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="2. CLAHE (ảnh trung gian)", use_container_width=True)
third.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="3. Kết quả YOLO", use_container_width=True)

if accepted:
    st.subheader("4. ROI cắt theo bounding box YOLO")
    roi_columns = st.columns(len(accepted))
    for item, column in zip(accepted, roi_columns):
        column.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB), caption=f"ROI #{item['number']}", use_container_width=True)

    st.subheader("5. HOG + histogram HSV → SVM phân loại")
    hog_columns = st.columns(len(accepted))
    for item, column in zip(accepted, hog_columns):
        column.image(hog_visualization(item["crop"]), caption=f"#{item['number']}: {item['name']}", use_container_width=True, clamp=True)

    st.subheader("6. HSV + Canny xác minh từng ROI")
    for item in accepted:
        hsv_column, canny_column = st.columns(2)
        verification = item["verification"]
        hsv_column.image(verification["mask"], caption=f"ROI #{item['number']} — mask HSV", use_container_width=True, clamp=True)
        canny_column.image(verification["edges"], caption=f"ROI #{item['number']} — cạnh Canny", use_container_width=True, clamp=True)

if detected:
    st.success("; ".join(detected))
else:
    st.warning("Không tìm thấy biển báo đạt ngưỡng tin cậy. Hãy thử hạ ngưỡng hoặc dùng ảnh rõ hơn.")
if rejected:
    st.info("Các dự đoán confidence thấp bị HSV/Canny loại: " + "; ".join(rejected))
