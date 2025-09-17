import torch
import torch.nn as nn
import torch.nn.functional as F


class Trend_aware_attention(nn.Module):

    def __init__(self, K, d, kernel_size):
        super(Trend_aware_attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D,D)
        self.FC = nn.Linear(D,D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size-1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)
    def forward(self, X):
        batch_size = X.shape[0]
        X_ = X.permute(0, 3, 2, 1)
        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        value = self.FC_v(X)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = (query @ key) * (self.d ** -0.5)
        attention = F.softmax(attention, dim=-1)
        X = (attention @ value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        return X.permute(0, 2, 1, 3)
