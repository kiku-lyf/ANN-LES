import numpy as np
from mpi4py import MPI
from numba import njit
import sys
import os

# Try importing optional libraries for rank 0
try:
    import openpyxl
    import pandas as pd
except ImportError:
    pass

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
Q = 9
NX = 1023
NY = 1023
U = 0.1

# LBM Parameters
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])

dx = 1.0
dy = 1.0
Lx_phys = dx * NY
Ly_phys = dy * NX
dt = dx
c = dx / dt
rho0 = 1.0
Re = 10000
niu = U * Lx_phys / Re
tau_f = 3.0 * niu + 0.5

# Domain Decomposition (Split Y dimension)
Global_X = NX + 1
Global_Y = NY + 1

if Global_Y % size != 0:
    if rank == 0:
        print(f"Error: NY+1 ({Global_Y}) must be divisible by size ({size}).")
    sys.exit(1)

Lx = Global_X          # Local X size (Full width)
Ly = Global_Y // size  # Local Y size (Slice height)
LYv = Ly + 2           # Local Y size with ghosts

# Arrays (Local)
mpi_rho = np.ones((Lx, LYv))
mpi_u = np.zeros((Lx, LYv, 2))
mpi_f = np.zeros((Lx, LYv, Q))
mpi_F = np.zeros((Lx, LYv, Q))
mpi_u0 = np.zeros((Lx, LYv, 2))

# Buffers for communication
# Layout: [F (Q), U (2), Rho (1)] * Lx
# We need to exchange 2 rows: 
# Send Top Row (local index Ly) -> Neighbor Above
# Send Bottom Row (local index 1) -> Neighbor Below
buffer_size = (Q + 2 + 1) * Lx
send_top = np.zeros(buffer_size)
send_btm = np.zeros(buffer_size)
recv_top = np.zeros(buffer_size)
recv_btm = np.zeros(buffer_size)

# Setup Output on Rank 0
if rank == 0:
    try:
        df = pd.DataFrame()
        df.to_excel('.train.xlsx', index=False)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        wb.save("endijtrain_011_1w.xlsx")
    except:
        pass

@njit
def feq(k, rho, u):
    eu = e[k][0] * u[0] + e[k][1] * u[1]
    uv = u[0] * u[0] + u[1] * u[1]
    return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

@njit
def init_local(mpi_f, mpi_u, mpi_rho, Lx, Ly):
    # Initialize f based on u=0, rho=1
    for i in range(Lx):
        for j in range(1, Ly + 1):
            for k in range(Q):
                mpi_f[i][j][k] = feq(k, mpi_rho[i][j], mpi_u[i][j])

@njit
def pack_buffers(send_top, send_btm, mpi_f, mpi_u, mpi_rho, Lx, Ly):
    # Top Row is at index Ly
    # Bottom Row is at index 1
    idx = 0
    for i in range(Lx):
        # Top Row Data
        for k in range(Q):
            send_top[idx] = mpi_f[i][Ly][k]
            idx += 1
        for m in range(2):
            send_top[idx] = mpi_u[i][Ly][m]
            idx += 1
        send_top[idx] = mpi_rho[i][Ly]
        idx += 1
    
    idx = 0
    for i in range(Lx):
        # Bottom Row Data
        for k in range(Q):
            send_btm[idx] = mpi_f[i][1][k]
            idx += 1
        for m in range(2):
            send_btm[idx] = mpi_u[i][1][m]
            idx += 1
        send_btm[idx] = mpi_rho[i][1]
        idx += 1

@njit
def unpack_buffers(recv_top, recv_btm, mpi_f, mpi_u, mpi_rho, Lx, Ly):
    # Receive from Top Neighbor -> Goes to Top Ghost (Ly + 1)
    # Receive from Bottom Neighbor -> Goes to Bottom Ghost (0)
    
    idx = 0
    for i in range(Lx):
        for k in range(Q):
            mpi_f[i][Ly + 1][k] = recv_top[idx]
            idx += 1
        for m in range(2):
            mpi_u[i][Ly + 1][m] = recv_top[idx]
            idx += 1
        mpi_rho[i][Ly + 1] = recv_top[idx]
        idx += 1
        
    idx = 0
    for i in range(Lx):
        for k in range(Q):
            mpi_f[i][0][k] = recv_btm[idx]
            idx += 1
        for m in range(2):
            mpi_u[i][0][m] = recv_btm[idx]
            idx += 1
        mpi_rho[i][0] = recv_btm[idx]
        idx += 1

def mpi_communicate():
    # Pack
    pack_buffers(send_top, send_btm, mpi_f, mpi_u, mpi_rho, Lx, Ly)
    
    # Neighbors
    top_nbr = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    btm_nbr = rank - 1 if rank > 0 else MPI.PROC_NULL
    
    # Send Top to Top_Nbr (He receives in his Btm Buffer? No, he receives my Top as his Btm Ghost)
    # Wait, if I am Rank 0, I send my Top (Ly) to Rank 1.
    # Rank 1 receives it into his Bottom Ghost (0).
    # So Rank 1's recv_btm should be filled by Rank 0's send_top.
    
    reqs = []
    
    # Send Top to Top Neighbor
    if top_nbr != MPI.PROC_NULL:
        reqs.append(comm.Isend(send_top, dest=top_nbr, tag=10))
        reqs.append(comm.Irecv(recv_top, source=top_nbr, tag=20))
        
    # Send Bottom to Bottom Neighbor
    if btm_nbr != MPI.PROC_NULL:
        reqs.append(comm.Isend(send_btm, dest=btm_nbr, tag=20)) # Tag 20 matches Top Nbr's recv logic
        reqs.append(comm.Irecv(recv_btm, source=btm_nbr, tag=10)) # Tag 10 matches Btm Nbr's send logic
        
    MPI.Request.Waitall(reqs)
    
    # Unpack
    unpack_buffers(recv_top, recv_btm, mpi_f, mpi_u, mpi_rho, Lx, Ly)

@njit
def mpi_evolution(mpi_f, mpi_u, mpi_rho, mpi_F, mpi_u0, rank, size, Lx, Ly, U):
    LYv = Ly + 2
    
    # Internal Evolution (Collision + Streaming)
    # Range: 1..Lx-2 (Global X inner), 1..Ly (Local Y)
    # Using ghosts at 0 and Ly+1
    
    for i in range(1, Lx - 1):
        for j in range(1, Ly + 1):
            for k in range(Q):
                ip = i - e[k][0]
                jp = j - e[k][1]
                mpi_F[i][j][k] = mpi_f[ip][jp][k] + (feq(k, mpi_rho[ip][jp], mpi_u[ip][jp]) - mpi_f[ip][jp][k]) / tau_f

    # Macro Variables Update
    for i in range(1, Lx - 1):
        for j in range(1, Ly + 1):
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

    # Boundaries
    # Left/Right (X=0 and X=NX=Lx-1)
    for j in range(1, Ly + 1):
        # Right (X=Lx-1)
        mpi_rho[Lx-1][j] = mpi_rho[Lx-2][j]
        for k in range(Q):
             mpi_f[Lx-1][j][k] = feq(k, mpi_rho[Lx-1][j], mpi_u[Lx-1][j]) + mpi_f[Lx-2][j][k] - feq(k, mpi_rho[Lx-2][j], mpi_u[Lx-2][j])
             
        # Left (X=0)
        mpi_rho[0][j] = mpi_rho[1][j]
        for k in range(Q):
             mpi_f[0][j][k] = feq(k, mpi_rho[0][j], mpi_u[0][j]) + mpi_f[1][j][k] - feq(k, mpi_rho[1][j], mpi_u[1][j])

    # Top Wall (Global Y = NY) -> Rank size-1, Local j = Ly
    if rank == size - 1:
        j_top = Ly
        for i in range(Lx):
            mpi_u[i][j_top][0] = U
            mpi_rho[i][j_top] = mpi_rho[i][j_top - 1]
            for k in range(Q):
                mpi_f[i][j_top][k] = feq(k, mpi_rho[i][j_top], mpi_u[i][j_top]) + mpi_f[i][j_top-1][k] - feq(k, mpi_rho[i][j_top-1], mpi_u[i][j_top-1])

    # Bottom Wall (Global Y = 0) -> Rank 0, Local j = 1
    if rank == 0:
        j_btm = 1
        for i in range(Lx):
            mpi_rho[i][j_btm] = mpi_rho[i][j_btm + 1]
            for k in range(Q):
                mpi_f[i][j_btm][k] = feq(k, mpi_rho[i][j_btm], mpi_u[i][j_btm]) + mpi_f[i][j_btm+1][k] - feq(k, mpi_rho[i][j_btm+1], mpi_u[i][j_btm+1])

@njit
def error_local(mpi_u, mpi_u0, Lx, Ly, rank, size):
    temp1 = 0.0
    temp2 = 0.0
    
    # Determine valid range for error calc
    # Skip Global Y=0 and Global Y=NY? LBM.py loops 1..NY-1.
    # Start J: 1 if rank==0 else 1
    # End J: Ly-1 if rank==size-1 else Ly
    
    j_start = 2 if rank == 0 else 1
    j_end = Ly - 1 if rank == size - 1 else Ly + 1 # range is exclusive at end
    
    for i in range(1, Lx - 1):
        for j in range(j_start, j_end):
            temp1 += ((mpi_u[i][j][0] - mpi_u0[i][j][0]) ** 2 + (mpi_u[i][j][1] - mpi_u0[i][j][1]) ** 2)
            temp2 += (mpi_u[i][j][0] ** 2 + mpi_u[i][j][1] ** 2)
    return temp1, temp2

@njit
def tforce(T, u, Ub):
    # Original tforce from LBM.py (Runs on gathered global u)
    for i in range(1, 127):
        for j in range(1, 127):
            ujj = 0.0
            uii = 0.0
            uij = 0.0
            ui = 0.0
            uj = 0.0
            sumw = 0.0
            
            for i1 in range(8):
                for j1 in range(8):
                    if (i1 == 0 or i1 == 7) and (j1 == 0 or j1 == 7):
                        wgt = 0.25
                    elif i1 == 0 or i1 == 7 or j1 == 0 or j1 == 7:
                        wgt = 0.5
                    else:
                        wgt = 1.0
                    
                    ux = u[i*8+i1][j*8+j1][0]
                    uy = u[i*8+i1][j*8+j1][1]
                    
                    ui += ux * wgt
                    uj += uy * wgt
                    uij += ux * uy * wgt
                    uii += ux * ux * wgt
                    ujj += uy * uy * wgt
                    sumw += wgt
            
            ui_norm = ui / sumw
            uj_norm = uj / sumw
            uij_norm = uij / sumw
            uii_norm = uii / sumw
            ujj_norm = ujj / sumw
            
            T[i][j][0] = uii_norm - ui_norm * ui_norm
            T[i][j][1] = uij_norm - ui_norm * uj_norm
            T[i][j][2] = uij_norm - ui_norm * uj_norm
            T[i][j][3] = ujj_norm - uj_norm * uj_norm
            Ub[i][j][0] = ui_norm
            Ub[i][j][1] = uj_norm

@njit
def ronudU(Ub, A, Am):
    # Original ronudU from LBM.py
    for i in range(1, 127):
        for j in range(1, 127):
            if i < 126 and j < 126:
                A[i][j][0] = Ub[i][j + 1][0] - Ub[i][j][0]
                A[i][j][1] = Ub[i][j + 1][1] - Ub[i][j][1]
                A[i][j][2] = Ub[i + 1][j][0] - Ub[i][j][0]
                A[i][j][3] = Ub[i + 1][j][1] - Ub[i][j][1]
            elif i == 126 or j == 126:
                if j == 126:
                    A[i][j][0] = Ub[i][j][0] - Ub[i][j-1][0]
                    A[i][j][1] = Ub[i][j][1] - Ub[i][j-1][1]
                else:
                    A[i][j][0] = Ub[i][j + 1][0] - Ub[i][j][0]
                    A[i][j][1] = Ub[i][j + 1][1] - Ub[i][j][1]
                if i == 126:
                    A[i][j][2] = Ub[i][j][0] - Ub[i-1][j][0]
                    A[i][j][3] = Ub[i][j][1] - Ub[i-1][j][1]
                else:
                    A[i][j][2] = Ub[i + 1][j][0] - Ub[i][j][0]
                    A[i][j][3] = Ub[i + 1][j][1] - Ub[i][j][1]
            Am[i][j] = np.sqrt(A[i][j][0]**2 + A[i][j][1]**2 + A[i][j][2]**2 + A[i][j][3]**2)

def writefileE1(Ub, A, T):
    filename = "endijtrain_011_1w.xlsx"
    try:
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
    except Exception:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"

    for i in range(128):
        for j in range(128):
            omega_abs = np.abs(A[i][j][1] - A[i][j][2])
            S_xy = 0.5 * (A[i][j][1] + A[i][j][2])
            Sm = np.sqrt(A[i][j][0]**2 + 2*S_xy**2 + A[i][j][3]**2)
            
            row = [
                float(i / 128),
                float(j / 128),
                omega_abs,
                A[i][j][0] + A[i][j][3],
                np.sqrt(A[i][j][0]**2 + A[i][j][1]**2 + A[i][j][2]**2 + A[i][j][3]**2),
                Sm,
                A[i][j][0],
                A[i][j][1],
                A[i][j][2],
                A[i][j][3],
                T[i][j][0],
                T[i][j][1],
                T[i][j][2],
                T[i][j][3]
            ]
            ws.append(row)
    wb.save(filename)

def output(n, u):
    with open(f'cavity_{n}.dat', 'w') as out:
        out.write("Title=\"LBM Lid Driven Flow\"\n")
        out.write("VARIABLES=\"X\",\"Y\",\"U\",\"V\"\n")
        out.write(f"ZONE T=\"BOX\",I= {NX + 1},J= {NY + 1},F=POINT\n")
        for j in range(NY + 1):
            for i in range(NX + 1):
                out.write(f"{i / Lx_phys} {j / Ly_phys} {u[i][j][0]} {u[i][j][1]}\n")

def main():
    # Initialize
    init_local(mpi_f, mpi_u, mpi_rho, Lx, Ly)
    
    # Global Arrays for analysis (Rank 0 only)
    if rank == 0:
        u_gathered = np.zeros((Global_X, Global_Y, 2))
        T = np.zeros((128, 128, 4))
        Ub = np.zeros((128, 128, 2))
        A = np.zeros((128, 128, 4))
        Am = np.zeros((128, 128, 1))
    
    n = 0
    while True:
        # Communication
        mpi_communicate()
        
        # Evolution
        mpi_evolution(mpi_f, mpi_u, mpi_rho, mpi_F, mpi_u0, rank, size, Lx, Ly, U)
        
        # Periodic Check (e.g. every 1000 steps)
        if n % 1000 == 0:
            # Error Check
            loc_e1, loc_e2 = error_local(mpi_u, mpi_u0, Lx, Ly, rank, size)
            tot_e1 = comm.reduce(loc_e1, op=MPI.SUM, root=0)
            tot_e2 = comm.reduce(loc_e2, op=MPI.SUM, root=0)
            
            if rank == 0:
                err = np.sqrt(tot_e1) / (np.sqrt(tot_e2) + 1e-30)
                print(f"n={n}, error={err:.6e}")
                
                # Check for convergence or output
                if n % 20000 == 0 or (n >= 10000 and err < 1.0e-6) or n % 5000 == 0:
                    # Need to gather full U
                    # Receive from all ranks
                    # Local part: mpi_u[:, 1:Ly+1, :]
                    pass

            # Gather logic (Blocking)
            # Send local valid u (Lx, Ly, 2) to rank 0
            local_valid_u = mpi_u[:, 1:Ly+1, :].copy() # Extract internal rows
            # Gather at rank 0
            # Result will be list of arrays on rank 0
            gathered_list = comm.gather(local_valid_u, root=0)
            
            if rank == 0:
                # Concatenate along Y axis (axis 1)
                # Note: gather returns list ordered by rank
                # Rank 0 is Bottom, Rank size-1 is Top.
                # Correct order for concatenation.
                u_gathered = np.concatenate(gathered_list, axis=1)
                
                # Check point
                print(f"Center U: {u_gathered[NX//2][NY//2]}")

                if n % 20000 == 0:
                    output(n, u_gathered)
                
                if n % 5000 == 0:
                    tforce(T, u_gathered, Ub)
                    ronudU(Ub, A, Am)
                    writefileE1(Ub, A, T)
                
                if n == 500000 or (n >= 10000 and err < 1.0e-6):
                    break
            
            # Broadcast break condition? 
            # If rank 0 decides to break, must notify others.
            stop = False
            if rank == 0:
                if n == 500000 or (n >= 10000 and err < 1.0e-6):
                    stop = True
            
            stop = comm.bcast(stop, root=0)
            if stop:
                break

        n += 1

if __name__ == "__main__":
    main()
