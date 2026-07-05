import cv2
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


def extract_orb(image, n_features=500):
    """
    Trích đặc trưng ORB từ ảnh.

    Parameters
    ----------
    image : ndarray
        Ảnh đầu vào (BGR hoặc Gray).
    n_features : int
        Số lượng keypoints tối đa.

    Returns
    -------
    keypoints
    descriptors
    """

    # Chuyển sang ảnh xám nếu cần
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    orb = cv2.ORB_create(nfeatures=n_features)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors
def build_codebook(descriptor_list, k=100):

    all_descriptors = np.vstack(descriptor_list)

    print("\n========== BUILD CODEBOOK ==========")
    print("Tổng descriptor:", all_descriptors.shape)
    print("Số visual words:", k)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1024
    )

    kmeans.fit(all_descriptors)

    return kmeans
def image_to_bow_hist(descriptors, codebook):
    """
    Chuyển descriptor của một ảnh thành histogram Bag of Visual Words.

    Parameters
    ----------
    descriptors : ndarray
        Descriptor ORB của một ảnh.
    codebook : MiniBatchKMeans
        Codebook đã được huấn luyện.

    Returns
    -------
    hist : ndarray
        Histogram Bag of Words.
    """

    k = codebook.n_clusters

    hist = np.zeros(k, dtype=np.float32)

    if descriptors is None:
        return hist

    # Gán mỗi descriptor vào visual word gần nhất
    words = codebook.predict(descriptors)

    # Đếm số lần xuất hiện
    for word in words:
        hist[word] += 1

    # Chuẩn hóa histogram
    if hist.sum() > 0:
        hist /= hist.sum()

    return hist
def train_svm_bow(X_train, y_train):
    """
    Huấn luyện Linear SVM trên đặc trưng Bag of Words.

    Parameters
    ----------
    X_train : ndarray
        Ma trận đặc trưng BoW.
    y_train : ndarray
        Nhãn.

    Returns
    -------
    scaler
    svm
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    svm = LinearSVC(
        random_state=42,
        max_iter=10000
    )

    svm.fit(X_train, y_train)

    return scaler, svm