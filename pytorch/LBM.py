import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

# 创建一个空的DataFrame
df = pd.DataFrame()

# 将DataFrame写入Excel文件
df.to_excel('.train.xlsx', index=False)
wb = openpyxl.Workbook()

# 获取默认创建的工作表
ws = wb.active

# 将工作表重命名为"Data"
ws.title = "Data"
Q = 9
NX = 1023
NY = 1023
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


@njit
def tforce(T,u,Ub):
    # 8×8盒式滤波，计算滤波后的速度和亚格子应力
    # 边界点权重0.5，内部点权重1.0，角点权重0.25
    for i in range(1,127):
        for j in range(1,127):
            ujj=0.0
            uii = 0.0
            uij=0.0
            ui=0.0
            uj=0.0
            sumw = 0.0  # 权重和
            
            for i1 in range(8):
                for j1 in range(8):
                    # 确定权重：边界0.5，角点0.25，内部1.0
                    if (i1 == 0 or i1 == 7) and (j1 == 0 or j1 == 7):
                        wgt = 0.25  # 角点
                    elif i1 == 0 or i1 == 7 or j1 == 0 or j1 == 7:
                        wgt = 0.5   # 边界点
                    else:
                        wgt = 1.0   # 内部点
                    
                    ux = u[i*8+i1][j*8+j1][0]
                    uy = u[i*8+i1][j*8+j1][1]
                    
                    ui += ux * wgt
                    uj += uy * wgt
                    uij += ux * uy * wgt
                    uii += ux * ux * wgt
                    ujj += uy * uy * wgt
                    sumw += wgt
            
            # 归一化（使用实际权重和）
            ui_norm = ui / sumw
            uj_norm = uj / sumw
            uij_norm = uij / sumw
            uii_norm = uii / sumw
            ujj_norm = ujj / sumw
            
            # 计算亚格子应力：τ_ij = overline(u_i u_j) - ū_i ū_j
            T[i][j][0] = uii_norm - ui_norm * ui_norm   # txx
            T[i][j][1] = uij_norm - ui_norm * uj_norm   # txy
            T[i][j][2] = uij_norm - ui_norm * uj_norm   # tyx (对称)
            T[i][j][3] = ujj_norm - uj_norm * uj_norm   # tyy
            Ub[i][j][0] = ui_norm                       # ux~
            Ub[i][j][1] = uj_norm                       # uy~

@njit
def ronudU(Ub,A,Am):
    #梯度张量 A_ij = ∂_i u_j
    # 使用中心差分，边界使用单边差分
    for i in range(1,127):
        for j in range(1,127):
            # 内部点使用中心差分
            if i < 126 and j < 126:
                # Axx = ∂_x u_x = (u_x[i][j+1] - u_x[i][j-1]) / 2
                # 简化为中心差分：Axx = u_x[i][j+1] - u_x[i][j]
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
            # 计算梯度张量的模
            Am[i][j]=np.sqrt(A[i][j][0]*A[i][j][0]+A[i][j][1]*A[i][j][1]+A[i][j][2]*A[i][j][2]+A[i][j][3]*A[i][j][3])



@njit
def feq(k, rho, u):
    eu = e[k][0] * u[0] + e[k][1] * u[1]
    uv = u[0] * u[0] + u[1] * u[1]
    return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)


@njit
def evolution(rho, u, f, u0):
    F = np.zeros((NY + 1, NX + 1, Q))
    for i in range(1, NX):
        for j in range(1, NY):
            for k in range(Q):
                ip = i - e[k][0]
                jp = j - e[k][1]
                F[i][j][k] = f[ip][jp][k] + (feq(k, rho[ip][jp], u[ip][jp]) - f[ip][jp][k]) / tau_f
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



def writefileE1(Ub,A,T):
    filename = "endijtrain_011_1w.xlsx"

    try:
        # 尝试打开现有的Excel文件
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
    except FileNotFoundError:
        # 如果文件不存在，创建新的Excel文件并添加一个名为Data的工作表
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"

    max_row = ws.max_row if ws.max_row > 0 else 1

    for i in range(128):
        for j in range(128):
            # 根据PDF方法一，输入特征应该包括：
            # 1. |ω|: 旋度大小 = |A_xy - A_yx|
            # 2. θ: 散度 = A_xx + A_yy
            # 3. √(A_ij A_ij): 速度梯度张量模
            # 4. √(S_ij S_ij): 应变率张量模
            # 以及A_ij的各个分量
            omega_abs = np.abs(A[i][j][1] - A[i][j][2])  # 旋度大小 |ω|
            S_xy = 0.5 * (A[i][j][1] + A[i][j][2])  # 应变率张量的xy分量
            Sm = np.sqrt(A[i][j][0]*A[i][j][0] + S_xy*S_xy + S_xy*S_xy + A[i][j][3]*A[i][j][3])  # 应变率张量模
            
            row = [
                float(i / 128),  # x坐标
                float(j / 128),  # y坐标
                omega_abs,  # |ω|: 旋度大小（输入特征0）
                A[i][j][0] + A[i][j][3],  # θ: 散度（输入特征1）
                np.sqrt(A[i][j][0]*A[i][j][0]+A[i][j][1]*A[i][j][1]+A[i][j][2]*A[i][j][2]+A[i][j][3]*A[i][j][3]),  # √(A_ij A_ij)（输入特征2）
                Sm,  # √(S_ij S_ij)（输入特征3）
                A[i][j][0],  # A_xx（输入特征4）
                A[i][j][1],  # A_xy（输入特征5）
                A[i][j][2],  # A_yx（输入特征6）
                A[i][j][3],  # A_yy（输入特征7）
                T[i][j][0],  # τ_xx（输出）
                T[i][j][1],  # τ_xy（输出）
                T[i][j][2],  # τ_yx（输出）
                T[i][j][3]   # τ_yy（输出）
            ]
            ws.append(row)

    # 保存Excel文件
    wb.save(filename)

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
    T = np.zeros((128, 128,4))  # 记录亚格子应力张量
    Ub=np.zeros((128, 128,2))   #记录滤波后的粗格子速度
    A = np.zeros((128, 128, 4)) #记录速度梯度张量
    Am = np.zeros((128, 128, 1))  # 记录速度梯度张量的模

    n = 0
    while True:
        evolution(rho, u, f, u0)
        if n % 10000 == 0:
            err = error(u, u0)
            tforce(T, u,Ub)
            ronudU(Ub,A,Am)
            print(f"The {n}th computation result:")
            print(f"The u,v pf point (NX/2,NY/2) is: {u[NY // 2][NX // 2][0]}, {u[NY // 2][NX // 2][1]}")
            print(f"The max relative error of uv is: {err:.6e}")
            if n >= 10000:
                if n % 20000 == 0:
                    output(n, u)
                if n% 5000==0:
                    writefileE1(Ub,A,T)
                if n==500000:
                    break
                if err < 1.0e-6:

                    break
        n += 1


if __name__ == "__main__":
    main()

