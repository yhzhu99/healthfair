from aif360.datasets import AdultDataset

# 1. 加载数据集并分割为训练集和验证集
dataset = AdultDataset()
(dataset_orig_train, dataset_orig_val) = dataset.split([0.7], shuffle=True)

# 准备训练集的特征 (X) 和目标 (y)
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()
import torch
from torch import nn
from torch.optim import Adam


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


# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()

# Initialize the model and optimizer
model = VAE(input_dim=X_train.shape[1], latent_dim=2, hidden_dim=256)
optimizer = Adam(model.parameters())

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(X_train_tensor)
    loss = loss_function(recon_batch, X_train_tensor, mu, logvar)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
