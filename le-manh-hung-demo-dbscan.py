import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Dữ liệu đầu vào
points = np.array([
    [1, 1], [2, 1], [2.5, 1.5], [3, 2], [4, 2], [5, 2.5], [8, 8], [8.5, 8.5]
])

# Số lượng hàng xóm k = số lượng thuộc tính + 1 = 2 (knee point gợi ý từ phân tích elbow)
k = 2

# Tính khoảng cách k-nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k).fit(points)
distances, indices = nbrs.kneighbors(points)

# Lấy khoảng cách lớn nhất trong k láng giềng gần nhất cho mỗi điểm
distances = np.sort(distances[:, k-1])  # Chỉ lấy khoảng cách của láng giềng thứ k

# Vẽ đồ thị k-distance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(distances) + 1), distances, marker='o', linestyle='-')
plt.axhline(y=2.69, color='r', linestyle='dashed', label=f'Suggested Eps ≈ 2.69')
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-Distance")
plt.title(f"{k}-Distance Graph")
plt.legend()
plt.grid()
plt.show()

# Thực hiện phân cụm bằng DBSCAN với Eps = 2.69 và MinPts = k
clustering = DBSCAN(eps=2.69, min_samples=k).fit(points)
labels = clustering.labels_

# Vẽ biểu đồ minh họa kết quả phân cụm
plt.figure(figsize=(8, 5))
for label in set(labels):
    cluster_points = points[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}' if label != -1 else 'Noise')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering with Eps ≈ 2.69")
plt.legend()
plt.grid()
plt.show()