import pandas as pd
from sklearn.metrics import calinski_harabasz_score
import numpy as np
import torch


def generate_mask(seq_lens):
    """Generates a mask for the sequence.

    Args:
        seq_lens: [batch size]
        (max_len: int)

    Returns:
        mask: [batch size, max_len]
    """
    max_len = torch.max(seq_lens).to(seq_lens.device)
    mask = torch.arange(max_len).expand(len(seq_lens), max_len).to(seq_lens.device)
    mask = mask < seq_lens.unsqueeze(1)
    return mask

def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

model_names = ['AdaCare', 'Agent', 'GRASP', 'GRU', 'LSTM', 'MCGRU', 'MLP', 'RETAIN', 'RNN', 'StageNet', 'TCN',
               'Transformer']
results = []
for model_name in model_names:
    for fold_index in range(10):
        multitask_file_path = './logs/analysis/tjh-multitask-' + model_name + '-fold' + str(fold_index) + '-seed0.pkl'
        outcome_file_path = './logs/analysis/tjh-outcome-' + model_name + '-fold' + str(fold_index) + '-seed0.pkl'
        multitask_file = pd.read_pickle(multitask_file_path)
        outcome_file = pd.read_pickle(outcome_file_path)
        print(outcome_file)
        lens = multitask_file['lens']
        labels = outcome_file['labels']
        multitask_embeddings = multitask_file['embeddings']
        outcome_embeddings = outcome_file['embeddings']
        mask = generate_mask(lens)
        # 获取每个患者最后一次就诊的嵌入和标签
        last_multitask_embeddings = get_last_visit(multitask_embeddings,mask)
        last_outcome_embeddings = get_last_visit(outcome_embeddings,mask)
        # last_multitask_embeddings = [multitask_embeddings[i, l - 1, :] for i, l in enumerate(lens)]
        # last_outcome_embeddings = [outcome_embeddings[i, l - 1, :] for i, l in enumerate(lens)]
        # 初始化一个索引来跟踪labels中的位置
        idx = 0
        last_labels = []
        for l in lens:
            idx += l - 1  # 移动到当前患者的最后一次就诊
            last_labels.append(labels[idx][0])
            idx += 1  # 移动到下一个患者的第一次就诊
        print(last_labels)
        # 检查多任务和结果嵌入的Calinski-Harabasz得分
        ch_score_multitask = calinski_harabasz_score(last_multitask_embeddings.numpy(), last_labels)
        ch_score_outcome = calinski_harabasz_score(last_outcome_embeddings.numpy(), last_labels)

        # 保存每个模型和fold的结果
        results.append({
            'model': model_name,
            'fold': fold_index,
            'CH Score (Multitask)': ch_score_multitask,
            'CH Score (Outcome)': ch_score_outcome
        })

# 将结果转换为DataFrame
df_results = pd.DataFrame(results)
df_results.to_excel("tjh_ch_results.xlsx")
# 打印DataFrame
print(df_results)
