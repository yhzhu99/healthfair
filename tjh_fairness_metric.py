import pandas as pd
from fairness_metric import calculate_bias
import os

results = []
def list_files_in_directory(directory_path):
    with os.scandir(directory_path) as entries:
        return [entry.path for entry in entries if entry.is_file()]


def fairness_metric(A, B):
    y_pred = []
    y_true = []
    sex = []
    age = []
    indices = []
    index = 0
    for count in A['lens']:
        for _ in range(count):
            indices.append(index)
        index += 1

    for index in range(len(A['preds'])):
        y_pred.append(A['preds'][index])
        y_true.append(A['labels'][index][0])
        sex.append(B[indices[index]][0][0])
        age.append(B[indices[index]][0][1])
    return calculate_bias(y_pred, y_true, {'sex': sex, 'age': age},
                          {'sex': lambda x: x == 1, 'age': lambda x: x <= 0}, 0.5)



for fold_index in range(10):
    a_file_path = './logs/analysis/cdsl-outcome-GRU-fold' + str(fold_index) + '-seed0.pkl'
    b_file_path = 'cdsl/processed/fold_' + str(fold_index) + '/test_x.pkl'
    A = pd.read_pickle(a_file_path)
    B = pd.read_pickle(b_file_path)
    fairness_performance = fairness_metric(A, B)
    result = {
        'Fold': fold_index,
        'DI_Sex': abs(fairness_performance['di']['sex']),
        'AOD_Sex': abs(fairness_performance['aod']['sex']),
        'EOD_Sex': abs(fairness_performance['eod']['sex']),
        'SPD_Sex': abs(fairness_performance['spd']['sex']),
        'DI_Age': abs(fairness_performance['di']['age']),
        'AOD_Age': abs(fairness_performance['aod']['age']),
        'EOD_Age': abs(fairness_performance['eod']['age']),
        'SPD_Age': abs(fairness_performance['spd']['age'])
    }

    results.append(result)

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 设置Fold为index
df.set_index('Fold', inplace=True)

# 导出到Excel
df.to_excel("cdsl_fairness_results.xlsx")
