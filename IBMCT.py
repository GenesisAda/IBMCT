import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from matplotlib import pyplot as plt
import torch.fft
import torch.nn.functional as F
from models.CBAM_torch import CBAMBlock
from models.Trend_aware_self_attention import Trend_aware_attention
from models.TimesNet import TimesBlock
from mamba_ssm import Mamba
from models.CoordAttention import CoordAtt
from pytorch_wavelets import DWT1D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scalerx = MinMaxScaler()
scalery = MinMaxScaler()

data = pd.read_csv('./data.csv')
for i in range(12):
    temp = data.iloc[:, i]
    window_size = 4000
    temp = temp.ewm(span=window_size).mean()
    data.iloc[:, i] = temp

all = 12
ratio = 3
real = int((all / (ratio+1)) * ratio)
virtual = all - real


def data_read():

    train_X = data.iloc[:16000, :real].values
    train_X = train_X * 1000
    train_X = scalerx.fit_transform(train_X)
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])

    train_y = data.iloc[:16000, real:].values
    train_y = train_y * 1000
    train_y = scalery.fit_transform(train_y)

    test_X = data.iloc[16000:20000, :real].values
    test_X = test_X * 1000
    test_X = scalerx.transform(test_X)
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    test_y = data.iloc[16000:20000, real:].values
    test_y = test_y * 1000

    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = data_read()


"""
IB-Mamba
"""
class Inception1D(nn.Module):

    def __init__(self, in_channels, out_channels_total):
        super(Inception1D, self).__init__()

        ch1 = out_channels_total // 4
        ch2 = out_channels_total // 4
        ch3 = out_channels_total // 4
        ch4 = out_channels_total - (ch1 + ch2 + ch3)
        channels_per_path = [ch1, ch2, ch3, ch4]

        self.path1 = nn.Conv1d(in_channels, channels_per_path[0], kernel_size=1)

        self.path2_1 = nn.Conv1d(in_channels, channels_per_path[1], kernel_size=1)
        self.path2_2 = nn.Conv1d(channels_per_path[1], channels_per_path[1], kernel_size=3, padding='same')

        self.path3_1 = nn.Conv1d(in_channels, channels_per_path[2], kernel_size=1)
        self.path3_2 = nn.Conv1d(channels_per_path[2], channels_per_path[2], kernel_size=5, padding='same')

        self.path4_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv1d(in_channels, channels_per_path[3], kernel_size=1)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, in_channels, sequence_length]

        out1 = self.gelu(self.path1(x))

        out2 = self.gelu(self.path2_1(x))
        out2 = self.gelu(self.path2_2(out2))

        out3 = self.gelu(self.path3_1(x))
        out3 = self.gelu(self.path3_2(out3))

        out4 = self.path4_1(x)
        out4 = self.gelu(self.path4_2(out4))


        x = torch.cat([out1, out2, out3, out4], dim=1)
        x = self.dropout(x)
        return x


class BiMambaPlusBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        identity = x
        x_norm = self.norm(x)
        forward_output = self.forward_mamba(x_norm)
        x_reversed = torch.flip(x_norm, dims=[1])
        backward_output_reversed = self.backward_mamba(x_reversed)
        backward_output = torch.flip(backward_output_reversed, dims=[1])
        fused_output = torch.cat([forward_output, backward_output], dim=-1)
        projected_output = self.out_proj(fused_output)
        output = identity + projected_output
        return output


"""
Abandon
"""
class WaveletBlock(nn.Module):

    def __init__(self, in_channels, seq_len, out_channels_total):
        super(WaveletBlock, self).__init__()

        self.dwt = DWT1D(wave='haar', J=1, mode='symmetric')

        dwt_feature_size = seq_len * in_channels

        self.fc = nn.Linear(dwt_feature_size, out_channels_total)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 输入 x shape: [batch_size, in_channels, seq_len]
        # e.g., [B, 1, 10]

        yl, yh = self.dwt(x)
        # yl.shape: [B, 1, 5], yh[0].shape: [B, 1, 5]

        x_wavelet = torch.cat([yl, yh[0]], dim=2)
        # x_wavelet.shape: [B, 1, 10]

        x_flattened = x_wavelet.view(x_wavelet.size(0), -1)
        # x_flattened.shape: [B, 10]


        x_projected = self.fc(x_flattened)
        # x_projected.shape: [B, 512]

        out = self.dropout(self.gelu(x_projected))

        out = out.unsqueeze(2)
        return out


''''''''''''''''Main model'''''''''''
class AttentionModel(nn.Module):
    def __init__(self, input_dims=real, time_steps=1, lstm_units=512, filters=512):
        super().__init__()
        self.time_steps = time_steps

        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dims, filters, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(filters, filters, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1))

        self.inception_block1 = Inception1D(in_channels=input_dims, out_channels_total=filters)
        self.inception_block2 = Inception1D(in_channels=filters, out_channels_total=filters)

        self.wavelet_block = WaveletBlock(in_channels=1, seq_len=input_dims,
                                          out_channels_total=filters)

        # 时间序列
        self.timeblock1 = TimesBlock(seq_len=T,pred_len=T,top_k=5,d_model=N,d_ff=N,num_kernels=3).to(device)
        self.timeblock2 = TimesBlock(seq_len=T, pred_len=T, top_k=5, d_model=N, d_ff=N, num_kernels=3).to(device)
        self.timeblock3 = TimesBlock(seq_len=T, pred_len=T, top_k=5, d_model=N, d_ff=N, num_kernels=3).to(device)

        self.mamba1 = Mamba(d_model=filters,d_state=16,d_conv=4,expand=2).to("cuda")
        self.mamba2 = Mamba(d_model=filters, d_state=16, d_conv=4, expand=2).to("cuda")
        self.mamba3 = Mamba(d_model=filters, d_state=16, d_conv=4, expand=2).to("cuda")

        self.bi_mamba_stack = nn.Sequential(
            BiMambaPlusBlock(d_model=filters, d_state=16, d_conv=4, expand=2),
            BiMambaPlusBlock(d_model=filters, d_state=16, d_conv=4, expand=2)
        ).to(device)

        self.co = CoordAtt(inp=filters, oup=filters)

        self.channel = CBAMBlock(channel=filters, reduction=16, kernel_size=7).to(device)
        self.trend1 = Trend_aware_attention(K=8, d=64, kernel_size=3).to(device)

        self.fc1 = nn.Linear(T, lstm_units * time_steps)
        self.fc = nn.Linear(lstm_units * time_steps, int(all-real))

    def forward(self, x):

        ''' IB-Mamba blcok '''
        x = x.permute(0, 2, 1)
        x = self.inception_block1(x)
        x = self.inception_block2(x)

        x = x.permute(0, 2, 1)
        x = self.bi_mamba_stack(x)

        ''' Trend-Infused Coordinate Attention (TICA) mechanism '''
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        x = self.co(x)

        x = x.permute(0, 3, 2, 1)
        x = self.trend1(x)

        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


INPUT_DIMS = real
TIME_STEPS = 1
lstm_units = 512
filters = 512
Batch_size = 1024
T = 512
N = 1

model = AttentionModel(INPUT_DIMS, TIME_STEPS, lstm_units, filters).to(device)

train_dataset = TensorDataset(
    torch.tensor(train_X, dtype=torch.float32),
    torch.tensor(train_y, dtype=torch.float32))
test_dataset = TensorDataset(
    torch.tensor(test_X, dtype=torch.float32),
    torch.tensor(test_y, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=Batch_size)

# 训练配置
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

patience = 20
no_improve = 0
early_stop = False

def train():
    best_val_loss = float('inf')
    num_epochs = 500
    for epoch in range(num_epochs):

        train_size = int(0.75 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=Batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=Batch_size,
            shuffle=False
        )

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "h.pth")
            no_improve = 0
        else:
            no_improve += 1
            print(f"Validation loss did not improve. No improvement for {no_improve} epochs.")

        if no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            early_stop = True
            break


# train()


model.load_state_dict(torch.load('weight/KW51/3.pth'))
model.eval()

# h
pre_list = []
test_y_list = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        pre_list.append(outputs.cpu().numpy())
        test_y_list.append(targets.cpu().numpy())

pre = np.vstack(pre_list)
test_y_np = np.vstack(test_y_list)

pre = scalery.inverse_transform(pre)
pre = pre / 1000
test_y = test_y / 1000

MAE_h_total = 0
MSE_h_total = 0
similarity_h_total = 0

for i in range(virtual):

    MAE_h_i = mean_absolute_error(test_y[:, i], pre[:, i])
    MSE_h_i = ((test_y[:, i] - pre[:, i]) ** 2).mean()
    area = np.trapz(np.fabs(pre[:, i]), dx=1)
    total_area = np.trapz(np.fabs(test_y[:, i]), dx=1)
    similarity = (1 - (np.abs(area - total_area) / total_area)) * 100

    print(f"Similarity_h_i:", similarity)
    print('MAE_h_i:', MAE_h_i)
    print('MSE_h_i:', MSE_h_i)

    MAE_h_total = MAE_h_total + MAE_h_i
    MSE_h_total = MSE_h_total + MSE_h_i
    similarity_h_total = similarity_h_total + similarity

MAE_h_total = MAE_h_total / virtual
MSE_h_total = MSE_h_total / virtual
similarity_h_total = similarity_h_total / virtual

print(f"Similarity_h:", similarity_h_total)
print('MAE_h:', MAE_h_total)
print('MSE_h:', MSE_h_total)







