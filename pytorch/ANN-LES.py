from numba import njit
import torch
import torch.nn as nn
import numpy as np

Q = 9
NX = 127
NY = 127
U = 0.1

e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])

dx = 1.0
dy = 1.0
Lx = dx * NY
Ly = dy * NX
dt = dx
c = dx / dt
rho0 = 1.0
Re = 10000
niu = U * Lx / Re
tau_f = 3.0 * niu + 0.5

# 定义神经网络模型（与训练时相同的结构）
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
    
    def forward(self, x):
        return self.model(x)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model1 = SubgridStressModel().to(device)
    model1.load_state_dict(torch.load('1wend1.pth', map_location=device))
    model1.eval()

    model2 = SubgridStressModel().to(device)
    model2.load_state_dict(torch.load('1wend3.pth', map_location=device))
    model2.eval()

    model3 = SubgridStressModel().to(device)
    model3.load_state_dict(torch.load('1wend3.pth', map_location=device))
    model3.eval()

    model4 = SubgridStressModel().to(device)
    model4.load_state_dict(torch.load('1wend4.pth', map_location=device))
    model4.eval()

    model5 = SubgridStressModel().to(device)
    model5.load_state_dict(torch.load('1wend4.pth', map_location=device))
    model5.eval()
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
    print("Please ensure all model files (.pth) are in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

@njit
def feq(k, rho, u):
    eu = e[k][0] * u[0] + e[k][1] * u[1]  #遵循Einstein求和约定计算
    uv = u[0] * u[0] + u[1] * u[1]        #求矢量u的模
    return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

@njit  # 通过模型训练得到的亚格子应力值
# 根据PDF方法一，输入特征应该包括：
# 1. |ω|: 旋度大小
# 2. θ: 散度 (A_xx + A_yy)
# 3. √(A_ij A_ij): 速度梯度张量模
# 4. √(S_ij S_ij): 应变率张量模
# 以及A_ij的各个分量
# 注意：为了保持与现有模型的兼容性，保持8个输入特征
def getT(u, A, inPut):
    for i in range(128):
        for j in range(128):
            # 输入0: 旋度大小 |ω| = |∂_x u_y - ∂_y u_x| = |A_xy - A_yx|
            # 根据PDF方法一，这是第一个输入特征
            inPut[i][j][0] = np.abs(A[i][j][1] - A[i][j][2])
            
            # 输入1: 散度 θ = A_xx + A_yy = ∂_x u_x + ∂_y u_y
            inPut[i][j][1] = A[i][j][0] + A[i][j][3]
            
            # 输入2: 速度梯度张量模 √(A_ij A_ij)
            inPut[i][j][2] = np.sqrt(
                A[i][j][0] * A[i][j][0] + A[i][j][1] * A[i][j][1] + 
                A[i][j][2] * A[i][j][2] + A[i][j][3] * A[i][j][3])
            
            # 输入3: 应变率张量模 √(S_ij S_ij)
            # S_ij = (A_ij + A_ji) / 2，所以 S_xx = A_xx, S_yy = A_yy
            # S_xy = S_yx = (A_xy + A_yx) / 2
            S_xy = 0.5 * (A[i][j][1] + A[i][j][2])
            inPut[i][j][3] = np.sqrt(
                A[i][j][0] * A[i][j][0] + S_xy * S_xy + 
                S_xy * S_xy + A[i][j][3] * A[i][j][3])
            
            # 输入4-7: A_ij的各个分量
            inPut[i][j][4] = A[i][j][0]  # A_xx
            inPut[i][j][5] = A[i][j][1]  # A_xy
            inPut[i][j][6] = A[i][j][2]  # A_yx
            inPut[i][j][7] = A[i][j][3]  # A_yy
            
    # 处理异常值
    for i in range(1, 128):
        for j in range(1, 128):
            for k in range(8):
                if inPut[i][j][k] < 1e-12 and inPut[i][j][k] > -1e-12:
                    inPut[i][j][k] = 0
                if np.isnan(inPut[i][j][k]) or np.isinf(inPut[i][j][k]):
                    inPut[i][j][k] = 0

@njit
def getF(Tp):
    # 计算 F_i = ∂_j τ_ij
    # 对于二维：F_x = ∂_x τ_xx + ∂_y τ_xy
    #          F_y = ∂_x τ_yx + ∂_y τ_yy
    F_pre = np.zeros((128, 128, 2))
    for i in range(1, 127):
        for j in range(1, 127):
            # 内部点使用中心差分
            if i < 126 and j < 126:
                # F_x = ∂_x τ_xx + ∂_y τ_xy
                F_pre[i][j][0] = (Tp[i][j + 1][0] - Tp[i][j][0]) + (Tp[i + 1][j][1] - Tp[i][j][1])  # Fx
                # F_y = ∂_x τ_yx + ∂_y τ_yy
                F_pre[i][j][1] = (Tp[i][j + 1][2] - Tp[i][j][2]) + (Tp[i + 1][j][3] - Tp[i][j][3])  # Fy
            # 边界点使用单边差分
            elif i == 126 or j == 126:
                if j == 126:
                    # 右边界：使用后向差分
                    F_pre[i][j][0] = (Tp[i][j][0] - Tp[i][j - 1][0]) + (Tp[i + 1][j][1] - Tp[i][j][1])  # Fx
                    F_pre[i][j][1] = (Tp[i][j][2] - Tp[i][j - 1][2]) + (Tp[i + 1][j][3] - Tp[i][j][3])  # Fy
                else:
                    # 内部：使用前向差分
                    F_pre[i][j][0] = (Tp[i][j + 1][0] - Tp[i][j][0]) + (Tp[i + 1][j][1] - Tp[i][j][1])  # Fx
                    F_pre[i][j][1] = (Tp[i][j + 1][2] - Tp[i][j][2]) + (Tp[i + 1][j][3] - Tp[i][j][3])  # Fy
                if i == 126:
                    # 上边界：使用后向差分
                    F_pre[i][j][0] = (Tp[i][j + 1][0] - Tp[i][j][0]) + (Tp[i][j][1] - Tp[i - 1][j][1])  # Fx
                    F_pre[i][j][1] = (Tp[i][j + 1][2] - Tp[i][j][2]) + (Tp[i][j][3] - Tp[i - 1][j][3])  # Fy
    return F_pre

@njit
def ronudU(Ub, A):
    # 梯度张量 A_ij = ∂_i u_j
    # 使用中心差分，边界使用单边差分
    for i in range(1, 127):
        for j in range(1, 127):
            # 内部点使用中心差分
            if i < 126 and j < 126:
                A[i][j][0] = Ub[i][j + 1][0] - Ub[i][j][0]  # Axx = ∂_x u_x
                A[i][j][1] = Ub[i][j + 1][1] - Ub[i][j][1]  # Axy = ∂_x u_y
                A[i][j][2] = Ub[i + 1][j][0] - Ub[i][j][0]  # Ayx = ∂_y u_x
                A[i][j][3] = Ub[i + 1][j][1] - Ub[i][j][1]  # Ayy = ∂_y u_y
            # 边界点使用单边差分
            elif i == 126 or j == 126:
                if j == 126:
                    A[i][j][0] = Ub[i][j][0] - Ub[i][j-1][0]  # Axx
                    A[i][j][1] = Ub[i][j][1] - Ub[i][j-1][1]  # Axy
                else:
                    A[i][j][0] = Ub[i][j + 1][0] - Ub[i][j][0]  # Axx
                    A[i][j][1] = Ub[i][j + 1][1] - Ub[i][j][1]  # Axy
                if i == 126:
                    A[i][j][2] = Ub[i][j][0] - Ub[i-1][j][0]  # Ayx
                    A[i][j][3] = Ub[i][j][1] - Ub[i-1][j][1]  # Ayy
                else:
                    A[i][j][2] = Ub[i + 1][j][0] - Ub[i][j][0]  # Ayx
                    A[i][j][3] = Ub[i + 1][j][1] - Ub[i][j][1]  # Ayy


# 计算额外应力项
# 根据PDF公式：g_K* = (1 - 1/(2τ)) * ω_K * [(e_Ki - u_i)/c_s² + (e_Ki u_i)/c_s⁴ * e_Ki] * F_i
# 对于D2Q9模型：c_s² = 1/3, c_s⁴ = 1/9
@njit
def extern_force(k, u, rho, F0):
    c_s2 = 1.0 / 3.0  # 声速平方
    c_s4 = 1.0 / 9.0  # 声速四次方
    
    # 计算 e_Ki · u_i (点积)
    eu_dot = e[k][0] * u[0] + e[k][1] * u[1]
    
    # 计算 (e_Ki - u_i) · F_i
    uf = (e[k][0] - u[0]) * F0[0] + (e[k][1] - u[1]) * F0[1]
    
    # 计算 (e_Ki · u_i) / c_s⁴ * e_Ki · F_i
    # 注意：这里 e_Ki 是向量，需要分别计算 x 和 y 分量
    eu_term_x = eu_dot / c_s4 * e[k][0] * F0[0]
    eu_term_y = eu_dot / c_s4 * e[k][1] * F0[1]
    eu_term = eu_term_x + eu_term_y
    
    # 计算 uf / c_s²
    uf_term = uf / c_s2
    
    # 根据PDF公式
    force = (1.0 - 0.5 / tau_f) * rho * w[k] * (uf_term + eu_term)
    
    if np.isnan(force) or np.isinf(force):
        return 0
    return force


@njit
def evolution(rho, u, f, u0, F0):
    F = np.zeros((NY + 1, NX + 1, Q))
    for i in range(1, NX):
        for j in range(1, NY):
            for k in range(Q):
                ip = i - e[k][0]
                jp = j - e[k][1]
                F[i][j][k] = f[ip][jp][k] + (feq(k, rho[ip][jp], u[ip][jp]) - f[ip][jp][k]) / tau_f + extern_force(k, u[ip][jp], rho[ip][jp], F0[ip][jp])
    for i in range(1, NX):
        for j in range(1, NY):
            rho[i][j] = 0
            u0[i][j][0] = u[i][j][0]
            u0[i][j][1] = u[i][j][1]
            u[i][j][0] = 0
            u[i][j][1] = 0
            for k in range(Q):
                f[i][j][k] = F[i][j][k]
                rho[i][j] += f[i][j][k]
                u[i][j][0] += e[k][0] * f[i][j][k]
                u[i][j][1] += e[k][1] * f[i][j][k]
            u[i][j][0] /= rho[i][j]
            u[i][j][1] /= rho[i][j]
    for j in range(1, NY):
        for k in range(Q):
            rho[NX][j] = rho[NX - 1][j]
            f[NX][j][k] = feq(k, rho[NX - 1][j], u[NX][j]) + f[NX - 1][j][k] - feq(k, rho[NX - 1][j], u[NX - 1][j])
            rho[0][j] = rho[1][j]
            f[0][j][k] = feq(k, rho[1][j], u[0][j]) + f[1][j][k] - feq(k, rho[1][j], u[1][j])
    for i in range(NX + 1):
        for k in range(Q):
            rho[i][0] = rho[i][1]

            f[i][0][k] = feq(k, rho[i][0], u[i][0]) + f[i][1][k] - feq(k, rho[i][1], u[i][1])
            u[i][NY][0] = U
            rho[i][NY] = rho[i][NY - 1]
            f[i][NY][k] = feq(k, rho[i][NY], u[i][NY]) + f[i][NY - 1][k] - feq(k, rho[i][NY - 1], u[i][NY - 1])


def output(m, u):
    with open(f'cavity_{m}.dat', 'w') as out:
        out.write("Title=\"LBM Lid Driven Flow\"\n")
        out.write("VARIABLES=\"X\",\"Y\",\"U\",\"V\"\n")
        out.write(f"ZONE T=\"BOX\",I= {NX + 1},J= {NY + 1},F=POINT\n")
        for j in range(NY + 1):
            for i in range(NX + 1):
                out.write(f"{i / Lx} {j / Ly} {u[i][j][0]} {u[i][j][1]}\n")

@njit
def error(u, u0):
    temp1 = 0
    temp2 = 0
    for i in range(1, NY):
        for j in range(1, NX):
            temp1 += ((u[i][j][0] - u0[i][j][0]) ** 2 + (u[i][j][1] - u0[i][j][1]) ** 2)
            temp2 += (u[i][j][0] ** 2 + u[i][j][1] ** 2)
    temp1 = np.sqrt(temp1)
    temp2 = np.sqrt(temp2)
    return temp1 / (temp2 + 1e-30)


def main():
    rho = np.ones((NY + 1, NX + 1))
    u = np.zeros((NY + 1, NX + 1, 2))
    f = np.zeros((NY + 1, NX + 1, Q))
    u0 = np.zeros((NY + 1, NX + 1, 2))  # 将u0初始化为全零数组
    n = 0
    A = np.zeros((128, 128, 4))  # 记录速度梯度张量
    Ub = np.zeros((128, 128, 4))  # 存储预测的亚格子应力张量（4个分量）
    F0 = np.zeros((128, 128, 2))  # 存储亚格子应力产生的力
    T1 = np.zeros((128 * 128, 8))
    inPut = np.zeros((128, 128, 8))
    
    while True:
        evolution(rho, u, f, u0, F0)
        if n % 1000 == 0 and n >= 2000:
            ronudU(u, A)
            getT(u, A, inPut)
            for i in range(128):
                for j in range(128):
                    for k in range(8):
                        T1[i * 128 + j][k] = inPut[i][j][k]

            # 使用PyTorch模型进行预测
            T_tensor = torch.FloatTensor(T1).to(device)
            
            with torch.no_grad():
                pren3 = model5(T_tensor).cpu().numpy()
                pre1 = model1(T_tensor).cpu().numpy()
                pre2 = model2(T_tensor).cpu().numpy()
                pre3 = model3(T_tensor).cpu().numpy()
                pre4 = model4(T_tensor).cpu().numpy()
            
            pre1 = -np.exp(pre1)
            pre4 = -np.exp(pre4)
            pre2 = np.exp(pre2)
            pre3 = np.exp(pre3)
            
            for i in range(len(pre2)):
                if pren3[i] < 0:
                    pre2[i] = -pre2[i]
                    pre3[i] = -pre3[i]
            
            for i in range(128):
                for j in range(128):
                    Ub[i][j][0] = pre1[i * 128 + j]
                    Ub[i][j][1] = pre2[i * 128 + j]
                    Ub[i][j][2] = pre3[i * 128 + j]
                    Ub[i][j][3] = pre4[i * 128 + j]

            F0 = getF(Ub)

            if n >= 1000:
                if n % 5000 == 0:
                    output(n, u)
                    print(f"The {n}th computation result:")
                    print(f"The u,v pf point (NX/2,NY/2) is: {u[NY // 2][NX // 2][0]}, {u[NY // 2][NX // 2][1]}")
            if n == 500000:
                break

        n += 1


if __name__ == "__main__":
    main()

