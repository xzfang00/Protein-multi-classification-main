import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
num_samples = 12
num_clusters = 3
data = []


for i in range(num_clusters):
    center = np.random.uniform(0, 1, 3)
    data.append(np.random.randn(num_samples // num_clusters, 3) + center)


data = np.vstack(data)

# 设置参数
k = num_clusters
T = 10
batch_size = 3


indices = np.random.choice(data.shape[0], k, replace=False)
centers = data[indices]


t = 0


labels = np.zeros(data.shape[0], dtype=int) - 1  # 初始化为-1，表示未分配


labels[indices] = np.arange(k)


def plot_clusters(data, centers, labels, t):
    colors = ['black', 'purple', 'red']
    plt.figure(figsize=(3, 3))

    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i + 1}')
        plt.scatter(centers[i][0], centers[i][1], s=100, color=colors[i], marker='X', edgecolors='black')

    plt.scatter(data[labels == -1][:, 0], data[labels == -1][:, 1], s=10, color='lightgrey', label='Unassigned')
    plt.title(f'Iteration {t}')
    plt.legend()
    plt.show()


plot_clusters(data, centers, labels, t)


while t < T:

    batch_indices = np.random.choice(data.shape[0], batch_size, replace=False)
    batch_data = data[batch_indices]


    for i, sample in enumerate(batch_data):
        distances = np.linalg.norm(sample - centers, axis=1)
        labels[batch_indices[i]] = np.argmin(distances)


    for j in range(k):
        cluster_points = data[labels == j]
        if len(cluster_points) > 0:
            centers[j] = np.mean(cluster_points, axis=0)


    plot_clusters(data, centers, labels, t + 1)


    t += 1

# 输出结果
print("The final set of cluster centers:")
print(centers)
print("Cluster label for each sample:")
print(labels)
