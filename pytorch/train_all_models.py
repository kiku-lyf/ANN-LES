"""
批量训练所有模型的脚本
训练4个模型，分别对应4个输出列（T[0], T[1], T[2], T[3]）
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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

def train_model_for_column(output_col, model_name, num_epochs=30, batch_size=16, lr=0.0001):
    """训练单个输出列的模型"""
    print(f"\n{'='*60}")
    print(f"Training model for output column {output_col} -> {model_name}")
    print(f"{'='*60}")
    
    # 导入数据
    df = pd.read_excel('end1w.xlsx', 0)
    df = df.iloc[:, :]
    x = df.iloc[:, 2:-4].values
    y = df.iloc[:, output_col].values
    y0 = np.array(y)
    x0 = np.array(x)
    
    # 对数变换
    y_log = np.log(np.abs(y0))
    for i in range(len(y0)):
        if y[i] > 0:
            y_log[i] = -y_log[i]
    
    # 创建数据集和数据加载器
    dataset = StressDataset(x0, y_log.reshape(-1, 1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SubgridStressModel().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 保存模型
    torch.save(model.state_dict(), model_name)
    print(f'Model saved to {model_name}')
    
    return model, train_losses, val_losses

def main():
    # 定义要训练的模型
    # 列索引和对应的模型文件名
    models_to_train = [
        (-4, '1wend1.pth'),  # T[0] (txx)
        (-3, '1wend2.pth'),  # T[1] (txy) - 注意：原代码中model2和model3都用1wend3.pth
        (-2, '1wend3.pth'),  # T[2] (tyx)
        (-1, '1wend4.pth'),  # T[3] (tyy)
    ]
    
    print("Starting batch training of all models...")
    print(f"Total models to train: {len(models_to_train)}")
    
    for output_col, model_name in models_to_train:
        try:
            train_model_for_column(output_col, model_name)
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("All models training completed!")
    print("="*60)
    print("\nNote: In ANN-LES.py, model5 uses the same model as model4 (1wend4.pth)")
    print("You may need to train an additional model for sign prediction if needed.")

if __name__ == "__main__":
    main()

