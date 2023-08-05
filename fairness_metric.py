import numpy as np
import random


def calculate_di(y_pred, y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的DI值
    di_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 计算特权群体和非特权群体得到正向预测结果的概率
        privileged_positive_rate = np.mean(y_pred[privileged_indices] == 1)
        unprivileged_positive_rate = np.mean(y_pred[unprivileged_indices] == 1)

        # 计算并存储这个敏感属性的DI值
        di_dict[attr] = unprivileged_positive_rate / privileged_positive_rate

    return di_dict


def calculate_aod(y_pred, y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的AOD值
    aod_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 提取特权群体和非特权群体的预测结果和真实标签
        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]
        y_true_privileged = y_true[privileged_indices]
        y_true_unprivileged = y_true[unprivileged_indices]

        # 计算特权群体和非特权群体的真阳性率 (TPR) 和假阳性率 (FPR)
        TPR_privileged = np.mean((y_pred_privileged == 1) & (y_true_privileged == 1))
        FPR_privileged = np.mean((y_pred_privileged == 1) & (y_true_privileged == 0))
        TPR_unprivileged = np.mean((y_pred_unprivileged == 1) & (y_true_unprivileged == 1))
        FPR_unprivileged = np.mean((y_pred_unprivileged == 1) & (y_true_unprivileged == 0))

        # 计算并存储这个敏感属性的AOD值
        aod_dict[attr] = 0.5 * ((FPR_privileged - FPR_unprivileged) + (TPR_privileged - TPR_unprivileged))

    return aod_dict


def calculate_eod(y_pred, y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的EOD值
    eod_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 提取特权群体和非特权群体的预测结果和真实标签
        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]
        y_true_privileged = y_true[privileged_indices]
        y_true_unprivileged = y_true[unprivileged_indices]

        # 计算特权群体和非特权群体的真阳性率 (TPR)
        TPR_privileged = np.mean((y_pred_privileged == 1) & (y_true_privileged == 1))
        TPR_unprivileged = np.mean((y_pred_unprivileged == 1) & (y_true_unprivileged == 1))

        # 计算并存储这个敏感属性的EOD值
        eod_dict[attr] = TPR_privileged - TPR_unprivileged

    return eod_dict


def calculate_spd(y_pred,y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的SPD值
    spd_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 提取特权群体和非特权群体的预测结果
        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]

        # 计算特权群体和非特权群体的正例预测率
        PPR_privileged = np.mean(y_pred_privileged == 1)
        PPR_unprivileged = np.mean(y_pred_unprivileged == 1)

        # 计算并存储这个敏感属性的SPD值
        spd_dict[attr] = PPR_privileged - PPR_unprivileged

    return spd_dict



# def calculate_theil_index(y_pred, y_true,sensitive_dict, privileged_conditions):
#     # 创建一个字典来存储每个敏感属性的Theil指数
#     theil_index_dict = {}
#
#     # 计算预测值的总平均值
#     y_pred_mean = np.mean(y_pred)
#
#     # 对于每个敏感属性
#     for attr, condition in privileged_conditions.items():
#         # 提取这个敏感属性的值
#         sensitive_values = np.array(sensitive_dict[attr])
#
#         # 确定哪些样本属于特权群体和非特权群体
#         condition_array = np.array([condition(val) for val in sensitive_values])
#         privileged_indices = np.where(condition_array)[0]
#         unprivileged_indices = np.where(~condition_array)[0]
#
#         # 提取特权群体和非特权群体的预测结果
#         y_pred_privileged = y_pred[privileged_indices]
#         y_pred_unprivileged = y_pred[unprivileged_indices]
#
#         # 计算特权群体和非特权群体的Theil指数
#         theil_index_privileged = np.mean((y_pred_privileged / y_pred_mean) * np.log(y_pred_privileged / y_pred_mean))
#         theil_index_unprivileged = np.mean((y_pred_unprivileged / y_pred_mean) * np.log(y_pred_unprivileged / y_pred_mean))
#
#         # 计算并存储这个敏感属性的Theil指数，对特权和非特权组加权求和
#         theil_index_dict[attr] = (len(y_pred_privileged) * theil_index_privileged + len(y_pred_unprivileged) * theil_index_unprivileged) / len(y_pred)
#
#     return theil_index_dict



def calculate_bias(y_pred, y_true, sensitive_dict, privileged_conditions, threshold):

    y_true = np.array(y_true)
    y_pred_class = [1 if y >= threshold else 0 for y in y_pred]
    y_pred_class = np.array(y_pred_class)
    performance_dict = {}
    performance_dict['di'] = calculate_di(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['aod'] = calculate_aod(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['eod'] = calculate_eod(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['spd'] = calculate_spd(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    # performance_dict['theil_index'] = calculate_theil_index(y_pred_class, y_true, sensitive_dict, privileged_conditions)

    return performance_dict


# 使用上面的函数
y_pred = np.array([0.1, 0.6, 0.7, 0.3])  # 这应该是您的模型预测结果
y_true = np.array([0, 1, 1, 0])  # 这应该是实际的标签
sensitive_dict = {'gender': [0, 1, 1, 0], 'age': [60, 40, 20, 40]}  # 这是敏感属性字典

privileged_conditions = {'gender': lambda x: x == 0, 'age': lambda x: x < 50}  # 这是特权条件字典

threshold = 0.5

performance_dict = calculate_bias(y_pred, y_true, sensitive_dict, privileged_conditions, threshold)

print(performance_dict)
