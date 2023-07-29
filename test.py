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

# 初始化逻辑回归模型
logreg = LogisticRegression(solver='liblinear', random_state=1)

# 在训练数据上拟合模型
logreg.fit(X_train, y_train)

# 准备验证集的特征 (X) 和目标 (y)
X_val = dataset_orig_val.features
y_val = dataset_orig_val.labels.ravel()

# 2. 在验证集上进行预测并计算 gap 值
y_pred_prob = logreg.predict_proba(X_val)
# 我们取类别间最大的预测概率作为置信度
confidence = np.max(y_pred_prob, axis=1)
# 假设类别为 0 和 1，绝对 gap 可以如下计算
print(confidence, y_val)
gap = np.abs(confidence - y_val)
print(gap)
# 3. 对 gap 值进行 K-means 聚类
# 假设我们想找到 K 个簇
K = 3
# 为 sklearn 重塑 gap（期望的是2D数组）
gap_reshape = gap.reshape(-1, 1)
print(gap_reshape)
kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(gap_reshape)

# KMeans 的 labels_ 属性给出了每个样本属于的簇
clusters = kmeans.labels_

print(clusters)
import pandas as pd

# 找出属于第二簇的所有样本
cluster2_indices = np.where(clusters == 2)[0]
cluster2_samples = X_val[cluster2_indices]

# 将其转换为DataFrame，以便进行分析
cluster2_df = pd.DataFrame(cluster2_samples, columns=dataset_orig_val.feature_names)

# 对于逻辑回归模型，可以查看每个特征的系数
coef_df = pd.DataFrame(logreg.coef_.T, index=dataset_orig_val.feature_names, columns=['Coefficient'])

print(coef_df)
