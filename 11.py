import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
df = pd.read_excel('end1w.xlsx', 0)
df = df.iloc[:, :]
x = df.iloc[:, 2:-4].values
y = df.iloc[:, -2].values  # 将y转换为浮点数
y0 = np.array(y)
x0=np.array(x)
# 归一化
# scaler_x = MinMaxScaler(feature_range=(0.00000000000001, 1))
# x_scaled = scaler_x.fit_transform(x0)
x_log=x0
# 对数变换


# 归一化 y
# scaler_y = MinMaxScaler(feature_range=(0.00000000000001, 1))
# y_scaled = scaler_y.fit_transform(y0.reshape(-1, 1))
y_log = np.log(np.abs(y0))
for i in range(len(y0)):
    if y[i] > 0:
        y_log[i] = -y_log[i]

# 神经网络搭建
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', kernel_initializer='random_normal', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='tanh', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.0001)
)

# 训练模型
model.fit(x_log, y_log, validation_split=0.20, epochs=30, batch_size=16)

# 保存模型
model.save('1w1end4.keras')

# 反归一化函数
def inverse_normalize(scaler, data):
    return  scaler.inverse_transform(np.exp(data))

xp = df.iloc[-128*128:, 2:-4].values
yp = df.iloc[-128*128:, -2].values
xp0=np.array(xp)
yp0=np.array(yp)
#xp0_logs=scaler_x.transform(xp0)
#xp0_log=np.log(np.abs(xp0))
y_pre = model.predict(xp0)


#y_prin=scaler_y.inverse_transform(y_pre_in)


data1 = y_pre.reshape(128, 128)

# 创建云图
plt.figure(figsize=(8, 8))
plt.imshow(data1, cmap='coolwarm', interpolation='nearest', vmin=(-1e-5)*8, vmax=(1e-5)*10)
plt.colorbar()
plt.title('Cloud Map pre')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
