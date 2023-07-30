from aif360.datasets import AdultDataset
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np

# 1. 加载数据集并分割为训练集和验证集
dataset = AdultDataset()
(dataset_orig_train, dataset_orig_val) = dataset.split([0.7], shuffle=True)

# 准备训练集的特征 (X) 和目标 (y)
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()
# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=10, random_state=0)  # 我们想要10个簇
kmeans.fit(X_train)

# 获取每个数据点的簇标签
labels = kmeans.labels_
print(labels)
# 你可以查看每个簇的中心
centers = kmeans.cluster_centers_
print(centers)