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


def get_so2c_score(value):
    if value < 92:
        return 2
    else:
        return 0


# 读取CSV文件
csv_path = 'cdsl/processed/fold_0/train_raw.csv'
df = pd.read_csv(csv_path, nrows=0)  # 只读取0行，这样可以仅获取表头
# 将表头存放在数组中
headers = df.columns.tolist()

print(headers)
# 查找指定元素在headers中的下标
indices = {}
for item in ["Sex", "Age", "U -- UREA", "PCR -- PROTEINA C REACTIVA", "SO2C -- sO2c (Saturaci¢n de ox¡geno)"]:
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
results_df = pd.DataFrame(columns=['Fold', 'Accuracy', 'AUROC', 'AUPRC', 'F1', 'MinPSE', 'ES'])

for fold_index in range(10):
    y_pred = []
    y_true = []
    y_los_true = []
    los_info_path = 'cdsl_before_zscore/processed/fold_' + str(fold_index) + '/los_info.pkl'
    los_info = pd.read_pickle(los_info_path)
    threshold = los_info['threshold']
    feature_file_path = 'cdsl_before_zscore/processed/fold_' + str(fold_index) + '/test_x.pkl'
    patients_features = pd.read_pickle(feature_file_path)
    outcome_file_path = 'cdsl_before_zscore/processed/fold_' + str(fold_index) + '/test_y.pkl'
    patients_outcomes = pd.read_pickle(outcome_file_path)
    c_score = []
    for patient_features, patient_outcomes in zip(patients_features, patients_outcomes):
        patient_scores = []
        for patient_feature, patient_outcome in zip(patient_features, patient_outcomes):
            # 获取每个特征的得分
            sex = get_sex_score(patient_feature[indices['Sex'] - 6])
            age = get_age_score(patient_feature[indices['Age'] - 6])
            urea = get_urea_score(patient_feature[indices['U -- UREA'] - 6])
            crp = get_crp_score(patient_feature[indices['PCR -- PROTEINA C REACTIVA'] - 6])
            so2c = get_so2c_score(patient_feature[indices['SO2C -- sO2c (Saturaci¢n de ox¡geno)'] - 6])
            # print(patient_feature[indices['Sex'] - 6], patient_feature[indices['Age'] - 6],
            #       patient_feature[indices['U -- UREA'] - 6], patient_feature[indices['PCR -- PROTEINA C REACTIVA'] - 6],
            #       patient_feature[indices['SO2C -- sO2c (Saturaci¢n de ox¡geno)'] - 6])
            # 将所有得分累加
            total_points = sex + age + urea + crp + so2c
            patient_scores.append([total_points, mortality[total_points], sex, age, urea, crp, so2c])
            y_pred.append(mortality[total_points])
            y_true.append(patient_outcome[0])
            y_los_true.append(patient_outcome[1])
        c_score.append(patient_scores)

    # [:1]
    # los_info
    metrics = get_binary_metrics(preds=torch.tensor(y_pred, dtype=torch.float32), labels=torch.tensor(y_true),
                                 y_true_los=torch.tensor(y_los_true, dtype=torch.float32), threshold=threshold)
    # Add each fold's metrics to the DataFrame
    results_df.loc[fold_index] = [fold_index,
                                  metrics.get('accuracy', 0),
                                  metrics.get('auroc', 0),
                                  metrics.get('auprc', 0),
                                  metrics.get('f1', 0),
                                  metrics.get('minpse', 0),
                                  metrics.get('es', 0)]

# Save results to Excel file
results_df.to_excel('cdsl_classification_metrics.xlsx', index=False)

print("Results exported to Excel.")
