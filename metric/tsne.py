from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 随机生成patient_embeds和outcomes
num_samples = 1000
embedding_dim = 128

patient_embeds = np.random.rand(num_samples, embedding_dim)  # 假设嵌入在0到1之间
outcomes = np.random.randint(0, 2, num_samples)  # 随机生成0或1
projected = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(patient_embeds)
# 合并降维后的数据和outcomes
outcomes_expanded = np.expand_dims(outcomes, axis=1)
concatenated = np.concatenate([projected, outcomes_expanded], axis=1)

df = pd.DataFrame(concatenated, columns=['Component 1', 'Component 2', 'Outcome'])
df['Outcome'].replace({1: 'Dead', 0: 'Alive'}, inplace=True)

sns.scatterplot(data=df, x="Component 1", y="Component 2", hue="Outcome", style="Outcome", palette=["C2", "C3"],
                alpha=0.5)
plt.show()
