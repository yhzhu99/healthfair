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
kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')  # 我们想要10个簇
kmeans.fit(X_train)

# 获取每个数据点的簇标签
labels = kmeans.labels_
print(labels)
# 你可以查看每个簇的中心
centers = kmeans.cluster_centers_

import pandas as pd

# 创建一个DataFrame，包含特征和簇标签
df = pd.DataFrame(X_train, columns=dataset_orig_train.feature_names)
df['cluster'] = labels

# 按簇标签分组，然后计算每个特征的统计量
cluster_stats = df.groupby('cluster').agg(['mean', 'median', 'min', 'max', 'std'])
# 输出到Excel文件
cluster_stats.to_excel('cluster_stats.xlsx')


# 将类别特征的数值编码转换回原始类别
df['race'] = dataset_orig_train.encoded2raw(df['race'])
df['sex'] = dataset_orig_train.encoded2raw(df['sex'])

# 计算每个簇中每个类别的频率
race_counts = df.groupby('cluster')['race'].value_counts(normalize=True)
sex_counts = df.groupby('cluster')['sex'].value_counts(normalize=True)

print(race_counts)
print(sex_counts)
