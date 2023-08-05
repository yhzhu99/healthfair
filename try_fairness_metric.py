from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# 加载数据集
dataset = AdultDataset()

# 划分数据集
train, test = dataset.split([0.8], shuffle=True)

# 定义模型
model = make_pipeline(StandardScaler(), LogisticRegression())

# 获取数据集特征和标签
X_train = train.features
y_train = train.labels.ravel()

X_test = test.features
y_test = test.labels.ravel()

# 拟合模型并做出预测
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 创建测试集的副本，并将预测标签添加到副本中
test_pred = test.copy()
test_pred.labels = y_pred

# 定义特权和非特权组
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

# 创建BinaryLabelDatasetMetric实例
metric = BinaryLabelDatasetMetric(test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

# 计算并打印公平性指标
print("Disparate Impact (DI):", metric.disparate_impact())
print("Statistical Parity Difference (SPD):", metric.statistical_parity_difference())

# 创建ClassificationMetric实例
classification_metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

# 计算并打印更多的公平性指标
print("Equal Opportunity Difference (EOD):", classification_metric.equal_opportunity_difference())
print("Average Odds Difference (AOD):", classification_metric.average_odds_difference())
print("Theil Index:", classification_metric.theil_index())

