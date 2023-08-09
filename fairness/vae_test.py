import torch
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from aif360.datasets import AdultDataset


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load and split the dataset
dataset = AdultDataset()
(dataset_orig_train, dataset_orig_val) = dataset.split([0.7], shuffle=True)

# Prepare the features (X) and target (y) from the training set
X_train = dataset_orig_train.features

# Create a scaler object and fit it to the training data
scaler = MinMaxScaler()
scaler.fit(X_train)

# Normalize the training data
X_train_scaled = scaler.transform(X_train)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled).float().to(device)
print(X_train.shape[1])
# Initialize the model and optimizer
model = VAE(input_dim=X_train.shape[1], latent_dim=10, hidden_dim=1024).to(device)
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(X_train_tensor)
    loss = loss_function(recon_batch, X_train_tensor, mu, logvar)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
import matplotlib.pyplot as plt

# 将模型设置为评估模式
model.eval()

# 通过编码器部分传递输入数据
mu, logvar = model.encode(X_train_tensor)

# 计算潜在变量
z = model.reparameterize(mu, logvar)

# 将潜在变量转换为numpy数组
z = z.detach().cpu().numpy()

# 绘制潜在变量的散点图
plt.figure(figsize=(8, 6))
plt.scatter(z[:, 0], z[:, 1], s=1)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Latent space')
plt.show()
