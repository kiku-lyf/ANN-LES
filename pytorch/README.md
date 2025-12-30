# 基于神经网络的格子玻尔兹曼方法顶盖驱动流湍流大涡模拟程序 (PyTorch版本)

## 一、环境配置

### Python环境
- Python 3.7+
- Anaconda3 (推荐)

### 软件包依赖
```bash
pip install torch torchvision numpy numba pandas openpyxl matplotlib scikit-learn
```

主要依赖包版本：
- PyTorch >= 1.9.0
- numpy >= 1.24.3
- numba >= 0.57.1
- pandas >= 2.0.3
- openpyxl >= 3.0.10
- matplotlib >= 3.7.2
- scikit-learn >= 1.3.0

## 二、使用说明

### 1. 生成训练数据
使用 `LBM.py` 程序生成训练数据：
- 设置格子大小（如1024*1024）
- 设置雷诺数
- 设置文件写入名
- 设置终止迭代次数或误差值

运行命令：
```bash
python LBM.py
```

运行后会得到该雷诺数的训练集（Excel文件）。

### 2. 数据清理
使用 `delete.py` 程序去除训练集中的无意义数据：
- 设置读入文件名和输出文件名

运行命令：
```bash
python delete.py
```

### 3. 训练神经网络

有两种方式训练神经网络：

#### 方式1：训练单个模型
使用 `train_model.py` 程序训练单个神经网络：
- 修改脚本中的 `output_col` 参数来选择不同的输出列（-4, -3, -2, -1）
- 分割数据的输入有多个，可以调节为带坐标的（x, y坐标）和不带坐标的
- 分割数据的输出只有一个
- 注意注释中的归一化操作可以选择使用或者不使用

运行命令：
```bash
python train_model.py
```

#### 方式2：批量训练所有模型（推荐）
使用 `train_all_models.py` 程序一次性训练所有需要的模型：
- 自动训练4个模型，分别对应4个输出列
- 更高效，避免重复加载数据

运行命令：
```bash
python train_all_models.py
```

训练好的模型会保存为 `.pth` 文件（PyTorch模型格式）：
- `1wend1.pth`: 对应 T[0] (txx)
- `1wend2.pth`: 对应 T[1] (txy)
- `1wend3.pth`: 对应 T[2] (tyx)
- `1wend4.pth`: 对应 T[3] (tyy)

### 4. 使用训练好的模型进行大涡模拟
使用 `ANN-LES.py` 程序：
- 在程序中读入训练好的模型（`.pth` 文件）
- 运行得到流场数据

运行命令：
```bash
python ANN-LES.py
```

### 5. 可视化
使用 Tecplot 软件读取流场数据（`.dat` 文件）得到流场图。

## 三、与TensorFlow版本的主要区别

1. **模型格式**：
   - TensorFlow版本：`.keras` 文件
   - PyTorch版本：`.pth` 文件

2. **模型加载**：
   - TensorFlow版本：`tf.keras.models.load_model()`
   - PyTorch版本：`torch.load()` 和 `model.load_state_dict()`

3. **模型预测**：
   - TensorFlow版本：`model.predict()`
   - PyTorch版本：`model(tensor)` 配合 `torch.no_grad()`

4. **训练代码**：
   - TensorFlow版本：使用 `tf.keras` API
   - PyTorch版本：使用 `torch.nn` 和 `torch.optim` API

## 四、文件说明

- `LBM.py`: 格子玻尔兹曼方法主程序，生成训练数据
- `delete.py`: 数据清理脚本
- `train_model.py`: 神经网络训练脚本（原11.py的PyTorch版本），用于训练单个模型
- `train_all_models.py`: 批量训练所有模型的脚本（推荐使用）
- `ANN-LES.py`: 使用训练好的神经网络进行大涡模拟的主程序
- `README.md`: 本说明文档

## 五、注意事项

1. 确保所有模型文件（`.pth`）与训练时使用的模型结构一致
2. 如果使用GPU加速，确保安装了CUDA版本的PyTorch
3. 数据文件路径需要根据实际情况调整
4. 模型文件名需要与代码中的文件名对应

