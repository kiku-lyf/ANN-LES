import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 定义神经网络模型
class SubgridStressModel(nn.Module):
    def __init__(self, input_size=8, hidden_sizes=[64, 32, 16], output_size=1):
        super(SubgridStressModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

# 自定义数据集
class StressDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 导入数据
df = pd.read_excel('data.xlsx', 0)
df = df.iloc[:, :]
x = df.iloc[:, 2:-4].values
# 可以根据需要选择不同的输出列
# -4: T[0] (txx), -3: T[1] (txy), -2: T[2] (tyx), -1: T[3] (tyy)
# 默认训练最后一列，可以通过修改output_col参数来训练不同的列
output_col = -2  # 可以改为 -4, -3, -2, -1 来训练不同的输出
y = df.iloc[:, output_col].values  # 将y转换为浮点数
y0 = np.array(y)
x0 = np.array(x)

# 归一化
# scaler_x = MinMaxScaler(feature_range=(0.00000000000001, 1))
# x_scaled = scaler_x.fit_transform(x0)
x_log = x0

# 归一化 y
# scaler_y = MinMaxScaler(feature_range=(0.00000000000001, 1))
# y_scaled = scaler_y.fit_transform(y0.reshape(-1, 1))
y_log = np.log(np.abs(y0))
for i in range(len(y0)):
    if y[i] > 0:
        y_log[i] = -y_log[i]

# 创建数据集和数据加载器
dataset = StressDataset(x_log, y_log.reshape(-1, 1))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SubgridStressModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 30
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# 保存模型
# 根据输出列保存不同的模型文件名
model_names = {-4: '1wend1.pth', -3: '1wend2.pth', -2: '1wend3.pth', -1: '1wend4.pth'}
model_filename = model_names.get(output_col, f'1wend{abs(output_col)}.pth')
torch.save(model.state_dict(), model_filename)
print(f'Model saved to {model_filename}')
print(f'Note: To train all models, run this script multiple times with different output_col values: -4, -3, -2, -1')

# 预测和可视化
xp = df.iloc[-128*128:, 2:-4].values
yp = df.iloc[-128*128:, -2].values
xp0 = np.array(xp)
yp0 = np.array(yp)

model.eval()
with torch.no_grad():
    xp_tensor = torch.FloatTensor(xp0).to(device)
    y_pre = model(xp_tensor).cpu().numpy()

# 反归一化函数
# def inverse_normalize(scaler, data):
#     return scaler.inverse_transform(np.exp(data))

data1 = y_pre.reshape(128, 128)

# 创建云图
plt.figure(figsize=(8, 8))
plt.imshow(data1, cmap='coolwarm', interpolation='nearest', vmin=(-1e-5)*8, vmax=(1e-5)*10)
plt.colorbar()
plt.title('Cloud Map pre')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

