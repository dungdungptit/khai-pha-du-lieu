import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Dữ liệu quốc gia
countries = np.array([
    [1.5, 150, 10, 0.3],
    [2.0, 180, 9, 0.4],
    [2.2, 160, 10.5, 0.35],
    [3.0, 200, 8.5, 0.5],
    [3.5, 250, 7.5, 0.6],
    [10.0, 40, 5, 0.8],
    [50.0, 10, 2, 0.95],
    [55.0, 5, 1.5, 0.98]
])

# Chuẩn hóa dữ liệu để đảm bảo khoảng cách công bằng
scaler = StandardScaler()
countries_scaled = scaler.fit_transform(countries)

# Cấu hình DBSCAN với giá trị Eps và MinPts phù hợp
eps_value = 1.2  # Điều chỉnh epsilon theo dữ liệu đã chuẩn hóa
min_samples_value = 2  # Số điểm tối thiểu để tạo cụm

# Áp dụng DBSCAN
clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(countries_scaled)
labels = clustering.labels_

# Vẽ biểu đồ kết quả phân cụm
plt.figure(figsize=(8, 5))
unique_labels = set(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))
for label in unique_labels:
    cluster_points = countries_scaled[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}' if label != -1 else 'Noise')
plt.xlabel("Standardized GDP")
plt.ylabel("Standardized Population Density")
plt.title(f"DBSCAN Clustering on Country Data (Eps = {eps_value}, MinPts = {min_samples_value})")
plt.legend()
plt.grid()
plt.show()
