import cv2
from features import extract_orb

image = cv2.imread("data/cropped/cam_o_to/Screenshot 2026-07-05 113544_watershed_001.jpg")

if image is None:
    print("Không đọc được ảnh!")
    exit()

print("Kích thước ảnh:", image.shape)

cv2.imshow("Image", image)
cv2.waitKey(0)

kp, des = extract_orb(image)

print("Số keypoints:", len(kp))

if des is not None:
    print("Descriptor shape:", des.shape)
else:
    print("Không tìm thấy descriptor")