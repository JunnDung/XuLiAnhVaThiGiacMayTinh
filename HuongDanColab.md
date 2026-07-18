# Huấn luyện YOLO một lớp trên Google Colab

YOLO trong project này chỉ dùng để tìm vị trí biển báo. Tên biển báo được HOG + HSV + SVM phân loại ở ứng dụng local.

1. Mở Google Colab và chọn **Runtime → Change runtime type → T4 GPU**.
2. Tải file ZIP project và `yolo_detector_update.zip` lên My Drive, sau đó mount Drive.
3. Giải nén project, chạy các lệnh sau trong thư mục project:

```python
!pip install -q ultralytics
!python ChuanBiYOLOMotLop.py
!python HuanLuyenYOLO.py --data vnts_detector.yaml --name vnts_detector --epochs 80 --imgsz 640 --batch 16 --device 0
```

4. Sau khi train xong, sao chép model về Drive:

```python
!cp "runs/vnts_detector/weights/best.pt" "/content/drive/MyDrive/best_detector.pt"
```

5. Tải `best_detector.pt` về máy, đặt vào:

```text
runs/vnts_detector/weights/best_detector.pt
```

6. Chạy ứng dụng:

```powershell
python -m streamlit run UngDungDuDoan.py
```
