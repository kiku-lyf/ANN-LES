import sys
import os
import time
from mpi4py import MPI
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Q = 9
NX = 63
NY = 63
U = 0.1

e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])

dx = 1.0
dy = 1.0
Lx = int(dx * (NX + 1))
Ly = int((dy * (NY + 1)) / size)
LXv = Lx  # 256
LYv = int(Ly + 2)  # 两行虚格点用于块间通信   256/size+2
dt = dx
c = dx / dt
rho0 = 1.0
Re = 400
niu = U * Lx / Re
tau_f = 3.0 * niu + 0.5

# dtype = np.float64

allsend = np.zeros(2 * (Q + 2 + 1) * Lx)
allReceive = np.zeros(2 * (Q + 2 + 1) * Lx)
# pdfsend=np.zeros(2*Q*Lx)
# pdfReceive=np.zeros(2*Q*Lx)
# pdusend=np.zeros(2*2*Lx)
# pduReceive=np.zeros(2*2*Lx)
# rhosend=np.zeros(2*Lx)
# rhoReceive=np.zeros(2*Lx)
rho = np.ones((NX + 1, NY + 1))
u = np.zeros((NX + 1, NY + 1, 2))
f = np.zeros((NX + 1, NY + 1, Q))
mpi_f = np.zeros((Lx, LYv, Q))
# mpi_F = np.zeros((Lx, LYv, Q))
mpi_u = np.zeros((Lx, LYv, 2))
# mpi_u0 = np.zeros((Lx, LYv, 2))
mpi_rho = np.zeros((Lx, LYv))
mpi_F = np.zeros((Lx, LYv, Q))
mpi_u0 = np.zeros((Lx, LYv, 2))


@njit
def feq(k, rho, u):
    eu = e[k][0] * u[0] + e[k][1] * u[1]
    uv = u[0] * u[0] + u[1] * u[1]
    return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)


@njit
def init(size, rank, f, u, mpi_f, mpi_u, mpi_rho):
    me = rank  # i号进程
    me_y = int(rank)  # 最后进程编号确定x，y方向最大编号来确定边界
    MY = size - 1
    # u0 = np.zeros((NY + 1, NX + 1, 2))  # 将u0初始化为全零数组
    # if(me==0):
    #     print(f"tau_f={tau_f}")
    for i in range(Lx):
        for j in range(me * Ly, (me + 1) * Ly):
            u[i][NY][0] = U
            for k in range(Q):
                f[i][j][k] = feq(k, rho[i][j], u[i][j])  # k, 9个方向分子数量

    # sys.stdout.flush()

    mpi_f[:, 1:LYv - 1] = f[:, int(me * Ly):int((me + 1) * Ly)]
    mpi_u[:, 1:LYv - 1] = u[:, int(me * Ly):int((me + 1) * Ly)]
    mpi_rho[:, 1:LYv - 1] = rho[:, int(me * Ly):int((me + 1) * Ly)]
    return me_y, MY


@njit
def befor_send(allsend, mpi_f, mpi_u, mpi_rho, Lx, Q, LYv):
    for i in range(Lx):
        for k in range(Q):
            # 将二维存到一维，上边界从0存，下边界从LX存
            allsend[i * Q + k] = mpi_f[i][LYv - 2][k]
            # pdfsend[i*Q+k]=mpi_f[i][LYv-2][k]#每个区域上边界
            # pdfsend[Lx*Q+i*Q+k]=mpi_f[i][1][k]#每个区域下边界
            allsend[Lx * Q + 3 * Lx + i * Q + k] = mpi_f[i][1][k]

        for m in range(2):
            allsend[Lx * Q + i * 2 + m] = mpi_u[i][LYv - 2][m]
            # pdusend[i*2+m]=mpi_u[i][LYv-2][m]
            # pdusend[Lx*2+i*2+m]=mpi_u[i][1][m]
            allsend[2 * Lx * Q + 3 * Lx + i * 2 + m] = mpi_u[i][1][m]
        allsend[Lx * Q + 2 * Lx + i] = mpi_rho[i][LYv - 2]
        allsend[2 * Lx * Q + 5 * Lx + i] = mpi_rho[i][1]
        # rhosend[i]=mpi_rho[i][LYv-2]
        # rhosend[Lx+i]=mpi_rho[i][1]


# allsend[:Lx * Q] = mpi_f[:, LYv - 2, :].reshape(-1)
# allsend[Lx * Q:Lx * Q + 2 * Lx] = mpi_u[:, LYv - 2, :].reshape(-1)
# allsend[Lx * Q + 2 * Lx:Lx * Q + 3 * Lx] = mpi_rho[:, LYv - 2]
# allsend[Lx * Q + 3 * Lx:2 * Lx * Q + 3 * Lx] = mpi_f[:, 1, :].reshape(-1)
# allsend[2 * Lx * Q + 3 * Lx:2 * Lx * Q + 5 * Lx] = mpi_u[:, 1, :].reshape(-1)
# allsend[2 * Lx * Q + 5 * Lx:2 * Lx * Q + 6 * Lx] = mpi_rho[:, 1]
def mpi_sendarecv(me_y, MY):
    # 子程序间信息相互传递的代码实现

    # 如果不是上边界和下边界，发送区域边界数据
    if (me_y != 0 and me_y != MY):
        # 向上发用tag=1，接收下面的tag=1
        # 想下发用tag=2，接收上面的tag=2
        allReceive[Lx * Q + 3 * Lx:] = comm.sendrecv(sendobj=allsend[:Lx * Q + 3 * Lx], dest=me_y + 1, sendtag=1,
                                                     recvbuf=None, source=me_y - 1, recvtag=1)
        allReceive[:Lx * Q + 3 * Lx] = comm.sendrecv(sendobj=allsend[Lx * Q + 3 * Lx:], dest=me_y - 1, sendtag=2,
                                                     recvbuf=None, source=me_y + 1, recvtag=2)

    # 下边界
    elif (me_y == 0):
        allReceive[:Lx * Q + 3 * Lx] = comm.sendrecv(sendobj=allsend[:Lx * Q + 3 * Lx], dest=me_y + 1, sendtag=1,
                                                     recvbuf=None, source=me_y + 1, recvtag=2)
    # 上边界
    else:
        allReceive[Lx * Q + 3 * Lx:] = comm.sendrecv(sendobj=allsend[Lx * Q + 3 * Lx:], dest=me_y - 1, sendtag=2,
                                                     recvbuf=None, source=me_y - 1, recvtag=1)


@njit
def after_receive(mpi_f, mpi_u, mpi_rho, allReceive, Lx, Q, LYv):
    for i in range(Lx):
        for k in range(Q):
            mpi_f[i, 0, k] = allReceive[Lx * Q + 3 * Lx + i * Q + k]
            mpi_f[i, LYv - 1, k] = allReceive[i * Q + k]

        for m in range(2):
            mpi_u[i, 0, m] = allReceive[2 * Lx * Q + 3 * Lx + i * 2 + m]
            mpi_u[i, LYv - 1, m] = allReceive[Lx * Q + i * 2 + m]

        mpi_rho[i, 0] = allReceive[2 * Lx * Q + 5 * Lx + i]
        mpi_rho[i, LYv - 1] = allReceive[Lx * Q + 2 * Lx + i]


@njit
def mpi_evolution(mpi_f, mpi_u, mpi_rho, mpi_F, mpi_u0, me_y, MY):
    # 对行进行划分，故所有逻辑先考虑不含上下边界情况
    # 如果不是上下边界,不需要进行上下边界处理
    # 添加两行新边界，然后对内部所有元素进行演化，得到的就是正确值，最后拼在一起即可

    # todo如果rank!=0&rank!=my
    if (me_y != 0 and me_y != MY):
        for i in range(1, NX):  # (1,Ly)原本10行，LYv=10+2=12行，序号0和序号11为用作传递的两行，在不考虑上下边界时需要演化1-10号，即（1,12-1）
            for j in range(1, LYv - 1):  # 假设size=2，LYv=256/2+2=130，演化不包括边界，0为新增下边界，129为新增上边界，故（1，129）
                for k in range(Q):
                    ip = i - e[k][0]  # x-direction pull migration
                    jp = j - e[k][1]  # y-direction migration
                    # rho[me_y*Ly+ip][jp],me_y号进程对应密度,u[me_y*Ly+ip][jp],me_y号进程对应速度
                    mpi_F[i][j][k] = mpi_f[ip][jp][k] + (
                                feq(k, mpi_rho[ip][jp], mpi_u[ip][jp]) - mpi_f[ip][jp][k]) / tau_f
                    # mpi_F[i][j][k] = decimal.Decimal(mpi_f[ip][jp][k]) + (feq(k, mpi_rho[ip][jp], mpi_u[ip][jp]) - decimal.Decimal(mpi_f[ip][jp][k])) / decimal.Decimal(tau_f)
        # 计算宏观量
        # !如果rank=0，则要从2开始,所以rank上边界和下边界处宏观量先不计算
        for i in range(1, NX):  # 256/4=64,LYv=64+2=66,(1,65)
            for j in range(1, LYv - 1):
                mpi_u0[i][j][0] = mpi_u[i][j][0]
                mpi_u0[i][j][1] = mpi_u[i][j][1]
                mpi_rho[i][j] = 0
                mpi_u[i][j][0] = 0
                mpi_u[i][j][1] = 0
                for k in range(Q):
                    mpi_f[i][j][k] = mpi_F[i][j][k]
                    mpi_rho[i][j] += mpi_f[i][j][k]
                    mpi_u[i][j][0] += e[k][0] * mpi_f[i][j][k]
                    mpi_u[i][j][1] += e[k][1] * mpi_f[i][j][k]
                mpi_u[i][j][0] /= mpi_rho[i][j]
                mpi_u[i][j][1] /= mpi_rho[i][j]

        # 左右边界,对于所有进程的左右边界处理都相同，从每个进程f[1][0][k]-f[LYv-2][0][k],f[1][NX][k]-f[LYv-2][NX][k]
        # !同理，上下边界处进程块的左右边界先不处理
        for j in range(1, LYv - 1):  # 每个进程1行到LYV-2行
            for k in range(Q):
                # 左边界，令rho0=rho1
                # 上下边界与左右边界重合部分在上下边界中计算
                mpi_rho[0][j] = mpi_rho[1][j]
                mpi_f[0][j][k] = feq(k, mpi_rho[0][j], mpi_u[0][j]) + mpi_f[1][j][k] - feq(k, mpi_rho[1][j],
                                                                                           mpi_u[1][j])
                mpi_rho[NX][j] = mpi_rho[NX - 1][j]
                mpi_f[NX][j][k] = feq(k, mpi_rho[NX][j], mpi_u[NX][j]) + mpi_f[NX - 1][j][k] - feq(k,
                                                                                                   mpi_rho[NX - 1][j],
                                                                                                   mpi_u[NX - 1][j])

    # todo如果rank=0
    # !要从2开始,0为传递消息行，1为下边界，从2开始演化

    if (me_y == 0):
        for i in range(1, NX):
            for j in range(2, LYv - 1):
                for k in range(Q):
                    ip = i - e[k][0]  # x-direction pull migration
                    jp = j - e[k][1]  # y-direction migration
                    mpi_F[i][j][k] = mpi_f[ip][jp][k] + (
                                feq(k, mpi_rho[ip][jp], mpi_u[ip][jp]) - mpi_f[ip][jp][k]) / tau_f

        for i in range(1, NX):  # 256/4=64,LYv=64+2=66,(1,65)
            for j in range(2, LYv - 1):
                mpi_u0[i][j][0] = mpi_u[i][j][0]
                mpi_u0[i][j][1] = mpi_u[i][j][1]

                mpi_rho[i][j] = 0
                mpi_u[i][j][0] = 0
                mpi_u[i][j][1] = 0
                for k in range(Q):
                    mpi_f[i][j][k] = mpi_F[i][j][k]
                    mpi_rho[i][j] += mpi_f[i][j][k]
                    mpi_u[i][j][0] += e[k][0] * mpi_f[i][j][k]
                    mpi_u[i][j][1] += e[k][1] * mpi_f[i][j][k]
                mpi_u[i][j][0] /= mpi_rho[i][j]
                mpi_u[i][j][1] /= mpi_rho[i][j]

        # !下边界处的左右边界处理，最下行左右边界留下
        for j in range(2, LYv - 1):
            for k in range(Q):
                # 左边界，令rho0=rho1
                # 上下边界与左右边界重合部分在上下边界中计算
                mpi_rho[0][j] = mpi_rho[1][j]
                mpi_f[0][j][k] = feq(k, mpi_rho[0][j], mpi_u[0][j]) + mpi_f[1][j][k] - feq(k, mpi_rho[1][j],
                                                                                           mpi_u[1][1])
                mpi_rho[NX][j] = mpi_rho[NX - 1][j]
                mpi_f[NX][j][k] = feq(k, mpi_rho[NX][j], mpi_u[NX][j]) + mpi_f[NX - 1][j][k] - feq(k,
                                                                                                   mpi_rho[NX - 1][j],
                                                                                                   mpi_u[NX - 1][j])
        # !下边界处理
        for i in range(0, NX + 1):
            for k in range(Q):
                mpi_rho[i][1] = mpi_rho[i][2]
                mpi_f[i][1][k] = feq(k, mpi_rho[i][1], mpi_u[i][1]) + mpi_f[i][2][k] - feq(k, mpi_rho[i][2],
                                                                                           mpi_u[i][2])

    # #todo 如果rank=MY
    # #!从1开始，到LYv-2(1,LYv-2)

    if (me_y == MY):
        for i in range(1, NX):
            for j in range(1, LYv - 2):  # 1-63
                for k in range(Q):
                    ip = i - e[k][0]  # x-direction pull migration
                    jp = j - e[k][1]  # y-direction migration
                    mpi_F[i][j][k] = mpi_f[ip][jp][k] + (
                                feq(k, mpi_rho[ip][jp], mpi_u[ip][jp]) - mpi_f[ip][jp][k]) / tau_f

        for i in range(1, NX):  # 256/4=64,LYv=64+2=66,(1,65)

            for j in range(1, LYv - 2):
                if (j == 63):
                    j = 63
                mpi_u0[i][j][0] = mpi_u[i][j][0]
                mpi_u0[i][j][1] = mpi_u[i][j][1]
                mpi_rho[i][j] = 0
                mpi_u[i][j][0] = 0
                mpi_u[i][j][1] = 0
                for k in range(Q):
                    mpi_f[i][j][k] = mpi_F[i][j][k]
                    mpi_rho[i][j] += mpi_f[i][j][k]
                    mpi_u[i][j][0] += e[k][0] * mpi_f[i][j][k]
                    mpi_u[i][j][1] += e[k][1] * mpi_f[i][j][k]
                mpi_u[i][j][0] /= mpi_rho[i][j]
                mpi_u[i][j][1] /= mpi_rho[i][j]

        # !左右边界处理，最上行左右边界留下

        for j in range(1, LYv - 2):
            # print(mpi_f[100][5],j)
            for k in range(Q):
                # 左边界，令rho0=rho1
                # 上下边界与左右边界重合部分在上下边界中计算
                mpi_rho[NX][j] = mpi_rho[NX - 1][j]

                mpi_f[NX][j][k] = feq(k, mpi_rho[NX][j], mpi_u[NX][j]) + mpi_f[NX - 1][j][k] - feq(k,
                                                                                                   mpi_rho[NX - 1][j],
                                                                                                   mpi_u[NX - 1][j])

                mpi_rho[0][j] = mpi_rho[1][j]
                mpi_f[0][j][k] = feq(k, mpi_rho[0][j], mpi_u[0][j]) + mpi_f[1][j][k] - feq(k, mpi_rho[1][j],
                                                                                           mpi_u[1][j])

        # !上边界处理

        for i in range(0, NX + 1):  # 1-255
            for k in range(Q):
                mpi_rho[i][LYv - 2] = mpi_rho[i][LYv - 3]
                mpi_u[i][LYv - 2][0] = U

                mpi_f[i][LYv - 2][k] = feq(k, mpi_rho[i][LYv - 2], mpi_u[i][LYv - 2]) + mpi_f[i][LYv - 3][k] - feq(k,
                                                                                                                   mpi_rho[
                                                                                                                       i][
                                                                                                                       LYv - 3],
                                                                                                                   mpi_u[
                                                                                                                       i][
                                                                                                                       LYv - 3])


def output(n, u):
    base_filename = f'cavity_{n}.dat'
    filename = base_filename
    file_index = 1

    # 检查文件是否存在
    while os.path.exists(filename):
        filename = f'cavity_{n}({file_index}).dat'
        file_index += 1
    with open(filename, 'w') as out:
        out.write("Title=\"LBM Lid Driven Flow\"\n")
        out.write("VARIABLES=\"X\",\"Y\",\"U\",\"V\"\n")
        out.write(f"ZONE T=\"BOX\",I= {NX + 1},J= {NY + 1},F=POINT\n")
        for i in range(1, NX + 1):
            for j in range(1, NY + 1):
                out.write(f"{i / (NX + 1)} {j / (NY + 1)} {u[i][j][0]} {u[i][j][1]}\n")


def mpi_gather():
    global all_udata, u
    mid_u = np.zeros((NX + 1, int(Ly), 2))
    mid_u[:, :] = mpi_u[:, 1:LYv - 1]
    if (rank == 0):
        all_udata = np.array([])

    else:
        all_udata = None
    all_udata = comm.gather(mid_u, root=0)
    if (rank == 0):
        all_udata_arrays = [np.array(sublist) for sublist in all_udata]

        u = np.concatenate(all_udata_arrays, axis=1)
        # end_time = time.time()
        comm.bcast(u, root=0)
    else:
        u = comm.bcast(None, root=0)


@njit
def error(mpi_u, mpi_u0, me_y, MY):
    temp1 = 0
    temp2 = 0
    if me_y == 0:
        temp1 += np.sum((mpi_u[1:NX, 2:Ly + 1] - mpi_u0[1:NX, 2:Ly + 1]) ** 2)
        temp2 += np.sum(mpi_u[1:NX, 2:Ly + 1] ** 2)
    elif me_y == MY:
        temp1 += np.sum((mpi_u[1:NX, 1:Ly] - mpi_u0[1:NX, 1:Ly]) ** 2)
        temp2 += np.sum(mpi_u[1:NX, 1:Ly] ** 2)
    else:
        temp1 += np.sum((mpi_u[1:NX, 1:Ly + 1] - mpi_u0[1:NX, 1:Ly + 1]) ** 2)
        temp2 += np.sum(mpi_u[1:NX, 1:Ly + 1] ** 2)
    return temp1, temp2


if __name__ == "__main__":

    start_time = time.time()
    me_y, MY = init(size, rank, f, u, mpi_f, mpi_u, mpi_rho)

    n = 0
    time4 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    time0 = 0
    while True:
        start_time0 = time.time()
        befor_send(allsend, mpi_f, mpi_u, mpi_rho, Lx, Q, LYv)
        end_time0 = time.time()
        time0 += end_time0 - start_time0
        start_time1 = time.time()
        mpi_sendarecv(me_y, MY)
        end_time1 = time.time()
        time1 += end_time1 - start_time1
        start_time4 = time.time()
        after_receive(mpi_f, mpi_u, mpi_rho, allReceive, Lx, Q, LYv)
        end_time4 = time.time()
        time4 += end_time4 - start_time4
        start_time2 = time.time()
        mpi_evolution(mpi_f, mpi_u, mpi_rho, mpi_F, mpi_u0, me_y, MY)
        end_time2 = time.time()
        time2 += end_time2 - start_time2
        # mid_u = np.zeros((NX+1, int(Ly)))

        if n % 100 == 0:
            start_time3 = time.time()
            temp_1, temp_2 = error(mpi_u, mpi_u0, me_y, MY)
            temp1 = comm.reduce(temp_1, op=MPI.SUM)
            temp2 = comm.reduce(temp_2, op=MPI.SUM)
            sys.stdout.flush()
            mpi_gather()
            end_time3 = time.time()
            time3 += end_time3 - start_time3
            if (rank == 0):
                temp1 = np.sqrt(temp1)
                temp2 = np.sqrt(temp2)
                err = temp1 / (temp2 + 1e-30)
                # print(f"The {n}th computation result:")
                # print(f"The u,v pf point (NX/2,NY/2) is: {u[NY // 2][NX // 2][0]}, {u[NY // 2][NX // 2][1]}")
                # print(f"The max relative error of uv is: {err:.6e}")
                if n >= 1000:
                    # if n==10000:
                    if err < 1.0e-6:
                        output(n, u)
                        end_time = time.time()

                        print(end_time - start_time)
                        print(time3, "all_erro_gather_time")
                        print(time0, "befor_send_time")
                        print(time1, "all_send_time")
                        print(time2, "all_evo_time")
                        print(time4, "after_recv_time")
                        sys.stdout.flush()

                        break
        n += 1
    x = np.linspace(0, 1, NX + 1)
    y = np.linspace(0, 1, NY + 1)
    X, Y = np.meshgrid(x, y)
    # u_array = np.array(u)
    u_t = np.transpose(u, (1, 0, 2))
    u_x = u_t[:, :, 0]
    u_y = u_t[:, :, 1]
    plt.figure(figsize=(6, 6))
    plt.streamplot(X, Y, u_x, u_y, density=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # plt.savefig("顶盖并行.png")
    plt.show()

# if __name__ == "__main__":
#     main()






