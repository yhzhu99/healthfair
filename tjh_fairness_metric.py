import pandas as pd
import numpy as np
from fairness_metric import calculate_bias

# 读取 csv 文件
df = pd.read_csv('test_after_zscore.csv')

# 提取 'Outcome', 'Sex', 'Age' 列，并将它们转换为 numpy 数组
outcome = df['Outcome'].values
sex = df['Sex'].values
age = df['Age'].values

# 假设 "outcome" 是你的目标数组
n_samples = len(outcome)  # 获取样本数量

# 生成随机的预测结果
y_pred = np.random.randint(0, 2, n_samples)  # 生成值为0或1的随机整数

# 现在 outcome, sex, age 是 numpy 数组，包含了对应列的值
print(outcome)
print(sex)
print(age)
performance_dict = calculate_bias(y_pred, outcome, {'sex': sex, 'age': age},
                                  {'sex': lambda x: x == 1, 'age': lambda x: x <= 0}, 0.5)
print(performance_dict)
