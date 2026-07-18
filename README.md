# Nhận diện Biển báo Giao thông Việt Nam

Bài tập lớn môn **Xử lý Ảnh và Thị giác Máy tính (121036)** — ĐH GTVT TP.HCM.

## Bài toán và giả thuyết H1

**Bài toán.** Phát hiện vị trí và phân loại tên biển báo giao thông Việt Nam trong ảnh đường phố thực tế thuộc bộ Vietnam Traffic Signs (VNTS).

**H1 — Pipeline lai CV cổ điển và học sâu.**

    Ảnh đường phố
      → CLAHE
      → YOLO một lớp: tìm bounding box traffic_sign
      → cắt ROI
      → HOG + histogram HSV
      → SVM: phân loại tên biển báo
      → HSV mask + Canny: kiểm tra màu và biên
      → kết quả cuối

YOLO một lớp chỉ trả lời **“biển báo ở đâu?”**. HOG + HSV + SVM trả lời **“đó là biển gì?”**.

| Kết quả | Tiêu chí kết luận H1 |
| --- | --- |
| Accuracy SVM ≥ 0.80 và Precision/Recall pipeline tốt trên test | H1 đạt |
| Chỉ một tiêu chí đạt | Đạt một phần |
| Các chỉ số đều thấp | Bác bỏ H1 |

## Dữ liệu VNTS

- 3.216 ảnh đường phố thực tế;
- 8.334 bounding box;
- 54 nhãn biển báo tiếng Việt;
- nhãn gốc YOLO: `class_id x_center y_center width height`.

```text
data/
  images/                 # ảnh VNTS gốc
  labels/                 # nhãn theo từng lớp biển báo
  classes_vie.txt         # tên lớp tiếng Việt
  classes_en.txt          # tên lớp tiếng Anh

data_detector/
  images/                 # ảnh cho YOLO một lớp
  labels/                 # mọi class được đổi thành 0 = traffic_sign
  splits/                 # train, val, test
```

Ảnh tự thêm để demo nhưng không có nhãn, ví dụ `TEST.jpg`, được tự động bỏ qua khi chuẩn bị dữ liệu train.

## Kỹ thuật áp dụng

| Chương | Kỹ thuật | Vai trò |
| --- | --- | --- |
| Ch. 2 | CLAHE | Tăng tương phản cục bộ. |
| Ch. 3 | HOG, Canny | Biểu diễn hình dạng, cạnh và cấu trúc ROI. |
| Ch. 4 | HSV mask + morphology | Phân đoạn màu đỏ, xanh, vàng. |
| Ch. 5 | YOLO11n một lớp | Định vị bounding box biển báo. |
| Ch. 5 | SVM | Phân loại tên biển từ HOG + HSV. |

YOLO được train với đúng một nhãn `traffic_sign`. Tên biển báo cuối cùng được tính bằng `classifier.predict(...)` của SVM, nên đây không phải dự án chỉ gọi YOLO đa lớp dạng hộp đen.

## Cấu trúc project

```text
.
├── data/                             # VNTS gốc
├── data_detector/                    # dữ liệu YOLO một lớp
├── runs/vnts_detector/weights/
│   └── best_detector.pt              # YOLO một lớp
├── reports/
│   ├── metrics.json                  # báo cáo SVM
│   ├── confusion_matrix.png          # ma trận nhầm lẫn SVM
│   └── svm_parameter_sweep.csv       # khảo sát SVM
├── cv_pipeline.py                    # CLAHE, HSV, Canny, HOG
├── ChuanBiYOLOMotLop.py              # tạo dữ liệu detector một lớp
├── HuanLuyenMoHinh.py                # train HOG + HSV + SVM
├── HuanLuyenYOLO.py                  # train YOLO một lớp
├── DanhGiaPipelineCV.py              # đánh giá Precision/Recall
├── KhaoSatThamSo.py                  # khảo sát tham số
├── UngDungDuDoan.py                  # ứng dụng Streamlit
├── main.ipynb                        # notebook thực nghiệm
├── vnts_detector.yaml
└── requirements.txt
```

## Cài đặt

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quy trình chạy

### 1. Chuẩn bị YOLO một lớp

```powershell
python ChuanBiYOLOMotLop.py
```

Script đổi toàn bộ nhãn biển báo thành class `0`, tạo train/val/test và cập nhật `vnts_detector.yaml` theo đường dẫn máy hiện tại.

### 2. Huấn luyện SVM phân loại tên biển

```powershell
python HuanLuyenMoHinh.py
```

    crop ROI theo nhãn gốc
      → resize 48 × 48
      → HOG: 9 orientations, cell 8 × 8
      → histogram HSV: 8 bins/kênh
      → StandardScaler + SVC RBF
      → thử C = 1, 10, 30

Kết quả được lưu:

```text
svm_model.joblib
reports/svm_parameter_sweep.csv
reports/metrics.json
reports/confusion_matrix.png
```

Kết quả hiện có: accuracy SVM khoảng **0.9064** trên tập kiểm tra tách ngẫu nhiên.

### 3. Huấn luyện YOLO một lớp

Nên dùng GPU Google Colab:

```powershell
python HuanLuyenYOLO.py --data vnts_detector.yaml --name vnts_detector --epochs 80 --imgsz 640 --batch 16 --device 0
```

Sau khi train, đặt model tại:

```text
runs/vnts_detector/weights/best_detector.pt
```

Xem [HuongDanColab.md](HuongDanColab.md) để train trên Colab.

### 4. Chạy ứng dụng

```powershell
python -m streamlit run UngDungDuDoan.py
```

App hiển thị: ảnh gốc, CLAHE, box YOLO, ROI, HOG, HSV mask, Canny, tên biển do SVM dự đoán, YOLO confidence và điểm CV.

### 5. Notebook

Mở [main.ipynb](main.ipynb), sau đó chọn **Run All**.

| Phần | Nội dung |
| --- | --- |
| 1 | Chạy pipeline trên ảnh mẫu |
| 2 | Sơ đồ pipeline và vai trò kỹ thuật |
| 3 | CLAHE, HSV, Canny với ba giá trị |
| 4 | HOG orientations 6/9/12 và SVM C = 1/10/30 |
| 5 | YOLO confidence 0.25/0.35/0.50 |
| 6 | Hướng dẫn rút kết luận thực nghiệm |

## Khảo sát tham số

```powershell
python KhaoSatThamSo.py --image data/images/0001.jpg
```

Tạo file `reports/parameter_sweep.png`.

| Kỹ thuật | Tham số | Ba giá trị |
| --- | --- | --- |
| CLAHE | clipLimit | 1.0, 2.0, 4.0 |
| HSV mask | saturation | 50, 70, 100 |
| Canny | ngưỡng thấp | 30, 60, 100 |
| HOG | orientations | 6, 9, 12 |
| SVM | C | 1, 10, 30 |
| YOLO | confidence | 0.25, 0.35, 0.50 |

## Đánh giá H1

```powershell
python DanhGiaPipelineCV.py
```

Script đánh giá tập test tại IoU = 0.5 và in Precision, Recall, TP, FP, FN cho pipeline:

```text
CLAHE → YOLO một lớp → HOG/HSV → SVM → HSV/Canny
```

Để báo cáo đầy đủ, lưu kết quả lần train 80 epoch từ Colab:

```text
results.csv, results.png, PR_curve.png, F1_curve.png,
P_curve.png, R_curve.png, confusion_matrix.png
```

Đặt chúng vào `reports/yolo_detector_80epochs/`.

## Hạn chế

- Biển báo nhỏ, mờ, che khuất hoặc ban đêm có thể bị bỏ sót.
- Lớp ít mẫu có thể bị SVM nhầm với biển hình dạng tương tự.
- Nên tách train/test theo ảnh gốc thay vì từng ROI để đánh giá nghiêm ngặt hơn.
- Có thể bổ sung ảnh tự chụp đã gán nhãn để tăng khả năng tổng quát.


