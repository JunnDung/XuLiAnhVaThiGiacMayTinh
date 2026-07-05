import os
import cv2
import numpy as np
import joblib
from pathlib import Path

# Import các hàm trích xuất đặc trưng từ module của bạn
from features import (
    extract_orb,
    build_codebook,
    image_to_bow_hist,
    train_svm_bow
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================================================================
# TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN QUẢN LÝ RIÊNG BIỆT (PROJECT ROOT RESOLUTION)
# =========================================================================
try:
    # Nếu chạy dạng file script .py thông thường
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    # Nếu copy đoạn code này vào chạy trên Jupyter Notebook
    CURRENT_DIR = Path(os.getcwd())

# Nếu file code của bạn nằm trong thư mục con riêng biệt (ví dụ: src, scripts, code, notebooks)
# Đoạn này giúp nhảy ngược ra thư mục gốc để tìm folder dữ liệu chính xác
if CURRENT_DIR.name in ["src", "scripts", "code", "notebooks", "features"]:
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR

DATASET_PATH = PROJECT_ROOT / "data" / "cropped"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 60)
print(" CẤU HÌNH HỆ THỐNG ĐƯỜNG DẪN")
print("=" * 60)
print(f" Thư mục gốc dự án: {PROJECT_ROOT}")
print(f" Thư mục chứa Dataset: {DATASET_PATH}")
print(f" Thư mục lưu trữ Models: {MODELS_DIR}")
print("=" * 60)

# Kiểm tra thư mục dữ liệu trước khi chạy
if not DATASET_PATH.exists():
    print(f"❌ [LỖI] Không tìm thấy thư mục dữ liệu tại: {DATASET_PATH}")
    print("Vui lòng kiểm tra lại vị trí các thư mục riêng biệt của bạn.")
    exit(1)

# =========================================================================
# ĐỌC DATASET ĐA THƯ MỤC
# =========================================================================
images = []
labels = []
class_names = []

print("\n" + "=" * 60)
print("BẮT ĐẦU ĐỌC DATASET")
print("=" * 60)

# Quét qua từng thư mục con (biencam, bienchidan, bienhieulenh, biennguyhiem)
for class_dir in sorted(DATASET_PATH.iterdir()):
    if class_dir.is_dir():
        class_name = class_dir.name

        # Tự động bỏ qua thư mục unknown theo logic thiết kế của bạn
        if class_name.lower() == "unknown":
            continue

        class_names.append(class_name)
        print(f"Đang đọc lớp: {class_name}")

        count = 0
        # Quét tất cả file ảnh trong thư mục lớp tương ứng
        for file_path in class_dir.glob("*"):
            if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Đọc ảnh sử dụng imdecode + np.fromfile giúp chống lỗi tuyệt đối
            # nếu đường dẫn thư mục riêng biệt của bạn chứa ký tự tiếng Việt có dấu
            image = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                print(f"   -> Không đọc được: {file_path.name}")
                continue

            images.append(image)
            labels.append(class_name)
            count += 1

        print(f"   -> Thành công: {count} ảnh")

# =========================================================================
# THỐNG KÊ DATASET
# =========================================================================
print("\n" + "=" * 60)
print("THỐNG KÊ DATASET HOÀN TẤT")
print("=" * 60)
print("Số lớp nhận diện:", len(class_names))
print("\nDanh sách các lớp:")
for i, name in enumerate(class_names):
    print(f"  {i + 1:02d}. {name}")
print("\nTổng số lượng ảnh hợp lệ đưa vào huấn luyện:", len(images))
print("=" * 60)

if len(images) == 0:
    print("❌ Không có ảnh nào được nạp. Dừng chương trình.")
    exit(1)

# =========================================================================
# TRÍCH XUẤT ORB DESCRIPTOR
# =========================================================================
print("\nĐang trích ORB descriptor...")
descriptor_list = []
valid_images = 0

for image in images:
    keypoints, descriptors = extract_orb(image)
    if descriptors is None:
        continue
    descriptor_list.append(descriptors)
    valid_images += 1

print(f"-> Số lượng ảnh trích xuất descriptor thành công: {valid_images}/{len(images)}")

# =========================================================================
# TẠO CODEBOOK (BAG OF VISUAL WORDS - KMEANS)
# =========================================================================
print("\nĐang khởi tạo Bag of Visual Words...")
codebook = build_codebook(
    descriptor_list=descriptor_list,
    k=100
)
print("\n Đã xây dựng Codebook thành công!")
print("=" * 60)

# =========================================================================
# BIỂU DIỄN ẢNH THÀNH VECTOR HISTOGRAM
# =========================================================================
print("\nĐang chuyển đổi toàn bộ dataset sang Vector Histogram Bag of Words...")
bow_features = []

for image in images:
    _, descriptors = extract_orb(image)
    hist = image_to_bow_hist(descriptors, codebook)
    bow_features.append(hist)

print(f"-> Đã tạo xong histogram cho {len(bow_features)} ảnh")
print(f"-> Kích thước đặc trưng mẫu (K visual words): {bow_features[0].shape}")

# =========================================================================
# ENCODE LABELS & CHUẨN BỊ DỮ LIỆU CHẠY SVM
# =========================================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X = np.array(bow_features)

print(f"\nSố lượng mẫu đặc trưng: {len(X)}")
print(f"Số lượng nhãn phân lớp mã hóa: {len(label_encoder.classes_)}")

# Chia tập dữ liệu Train / Test (Tỷ lệ 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  + Kích thước tập Train: {len(X_train)}")
print(f"  + Kích thước tập Test : {len(X_test)}")

# =========================================================================
# HUẤN LUYỆN TUYẾN TÍNH SUPPORT VECTOR MACHINE (LINEAR SVM)
# =========================================================================
print("\nĐang huấn luyện Linear SVM trên đặc trưng BoW...")
scaler, svm = train_svm_bow(X_train, y_train)
print(" Huấn luyện mô hình phân loại thành công!")

# =========================================================================
# ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH (EVALUATION)
# =========================================================================
X_test_scaled = scaler.transform(X_test)
pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print("\n" + "=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60)
print(f"Độ chính xác tổng thể (Accuracy): {round(acc * 100, 2)} %")
print("\nBáo cáo chi tiết các chỉ số (Classification Report):\n")
print(
    classification_report(
        y_test,
        pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)
print("=" * 60)

# =========================================================================
# LƯU TRỮ TRỌNG SỐ PIPELINE MÔ HÌNH VÀO THƯ MỤC MODELS RIÊNG BIỆT
# =========================================================================
# Tạo thư mục quản lý model riêng biệt nếu chưa tồn tại trên máy của bạn
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(codebook, MODELS_DIR / "orb_codebook.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(svm, MODELS_DIR / "svm_classifier.pkl")
joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

print(f"\n Đã xuất và lưu trữ thành công toàn bộ file weights vào thư mục: {MODELS_DIR}\n")