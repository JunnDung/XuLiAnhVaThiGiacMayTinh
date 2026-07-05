import os
import cv2
import joblib
import numpy as np
import argparse

from features import extract_orb, image_to_bow_hist

# ==============================
# Đọc model đã lưu
# ==============================

codebook = joblib.load("models/orb_codebook.pkl")
scaler = joblib.load("models/scaler.pkl")
svm = joblib.load("models/svm_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ==============================
# Đọc tham số dòng lệnh
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument(
    "image",
    help="Đường dẫn ảnh cần nhận diện"
)

args = parser.parse_args()
print("Đường dẫn ảnh:", args.image)
print("File tồn tại:", os.path.exists(args.image))

# ==============================
# Đọc ảnh
# ==============================

try:
    image = cv2.imdecode(
        np.fromfile(args.image, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
except Exception as e:
    print("Lỗi khi đọc ảnh:", e)
    exit()

if image is None:
    print("Không đọc được ảnh!")
    exit()

# ==============================
# Trích ORB
# ==============================

_, descriptors = extract_orb(image)

if descriptors is None:
    print("Không tìm thấy descriptor!")
    exit()

# ==============================
# Chuyển sang Bag of Words
# ==============================

hist = image_to_bow_hist(
    descriptors,
    codebook
)

hist = hist.reshape(1, -1)

# ==============================
# Chuẩn hóa
# ==============================

hist = scaler.transform(hist)

# ==============================
# Dự đoán
# ==============================

prediction = svm.predict(hist)

label = label_encoder.inverse_transform(prediction)

print("=" * 40)
print("Kết quả nhận diện:")
print(label[0])
print("=" * 40)

# ==============================
# Hiển thị ảnh
# ==============================

cv2.imshow("Input", image)
cv2.waitKey(0)
cv2.destroyAllWindows()