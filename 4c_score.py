import pandas as pd
from binary_classification_metrics import get_binary_metrics
import torch

sex_score = {0: 0, 1: 1}
age_score = {'18-49': 0, '50-59': 2, '60-69': 4, '70-79': 6, '80': 7}
urea_score = {'<7': 0, '7-14': 1, '>14': 3}
crp_score = {'<50': 0, '50-99': 1, '>=100': 2}
mortality = [0, 0.003, 0.008, 0.023, 0.048, 0.075, 0.078, 0.117, 0.144, 0.192, 0.229, 0.269, 0.329, 0.401, 0.446, 0.516,
             0.591, 0.661, 0.758, 0.774, 0.829, 0.875]


def get_sex_score(value):
    return sex_score.get(value, 0)


def get_age_score(value):
    for key, score in age_score.items():
        if key == '80' and value >= 80:
            return score
        elif '-' in key:
            start, end = map(int, key.split('-'))
            if start <= value <= end:
                return score
    return 0


def get_urea_score(value):
    if value < 7:
        return urea_score['<7']
    elif 7 <= value <= 14:
        return urea_score['7-14']
    else:
        return urea_score['>14']


def get_crp_score(value):
    if value < 50:
        return crp_score['<50']
    elif 50 <= value < 100:
        return crp_score['50-99']
    else:
        return crp_score['>=100']


# 读取CSV文件
csv_path = 'tjh/processed/fold_0/train_raw.csv'
df = pd.read_csv(csv_path, nrows=0)  # 只读取0行，这样可以仅获取表头

# 将表头存放在数组中
headers = df.columns.tolist()

print(headers)
# 查找指定元素在headers中的下标
indices = {}
for item in ["Sex", "Age", "Urea", "Hypersensitive c-reactive protein"]:
    try:
        indices[item] = headers.index(item)
    except ValueError:
        indices[item] = None

print(indices)
accuracy = []
auroc = []
auprc = []
f1 = []
minpse = []
# 创建一个空的DataFrame来存储指标
results_df = pd.DataFrame(columns=['Fold', 'Accuracy', 'AUROC', 'AUPRC', 'F1', 'MinPSE'])

for fold_index in range(10):
    y_pred = []
    y_true = []
    feature_file_path = 'cdsl_before_zscore/processed/fold_' + str(fold_index) + '/val_x.pkl'
    patients_features = pd.read_pickle(feature_file_path)
    outcome_file_path = 'cdsl_before_zscore/processed/fold_' + str(fold_index) + '/val_y.pkl'
    patients_outcomes = pd.read_pickle(outcome_file_path)
    c_score = []
    for patient_features, patient_outcomes in zip(patients_features, patients_outcomes):
        patient_scores = []
        for patient_feature, patient_outcome in zip(patient_features, patient_outcomes):
            # 获取每个特征的得分
            sex = get_sex_score(patient_feature[indices['Sex'] - 6])
            age = get_age_score(patient_feature[indices['Age'] - 6])
            urea = get_urea_score(patient_feature[indices['Urea'] - 6])
            crp = get_crp_score(patient_feature[indices['Hypersensitive c-reactive protein'] - 6])
            # 将所有得分累加
            total_points = sex + age + urea + crp
            patient_scores.append([total_points, mortality[total_points], sex, age, urea, crp])
            y_pred.append(mortality[total_points])
            y_true.append(patient_outcome[0])
        c_score.append(patient_scores)
    print(c_score)
    metrics = get_binary_metrics(preds=torch.tensor(y_pred, dtype=torch.float32), labels=torch.tensor(y_true))
    # Add each fold's metrics to the DataFrame
    results_df.loc[fold_index] = [fold_index,
                                  metrics.get('accuracy', 0),
                                  metrics.get('auroc', 0),
                                  metrics.get('auprc', 0),
                                  metrics.get('f1', 0),
                                  metrics.get('minpse', 0)]

# Calculate mean and standard deviation
mean_series = results_df.mean()
std_series = results_df.std()

# Add mean and standard deviation to the DataFrame
results_df.loc['Mean'] =mean_series.tolist()
results_df.loc['Std'] =std_series.tolist()

# Save results to Excel file
results_df.to_excel('cdsl_classification_metrics.xlsx', index=False)

print("Results exported to Excel.")