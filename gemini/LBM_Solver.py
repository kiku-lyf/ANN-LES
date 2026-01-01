import torch
import numpy as np

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

class LBMConfig:
    def __init__(self, nx, ny, Re, u_lid):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.u_lid = u_lid
        self.ly = ny - 1
        self.lx = nx - 1
        self.nu = u_lid * self.ly / Re
        self.tau = 3.0 * self.nu + 0.5
        self.omega = 1.0 / self.tau
        
        # D2Q9 参数
        self.w = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=DEVICE, dtype=DTYPE)
        self.cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=DEVICE, dtype=DTYPE)
        self.cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=DEVICE, dtype=DTYPE)
        self.cs_sq = 1/3

class LBMSolver:
    def __init__(self, config):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        
        # 初始化分布函数 f
        # f shape: [9, ny, nx]
        self.f = torch.zeros((9, self.ny, self.nx), device=DEVICE, dtype=DTYPE)
        self.rho = torch.ones((self.ny, self.nx), device=DEVICE, dtype=DTYPE)
        self.u = torch.zeros((2, self.ny, self.nx), device=DEVICE, dtype=DTYPE) # u[0]=ux, u[1]=uy
        
        # 初始化为顶盖驱动的平衡态
        self.u[0, -1, :] = self.cfg.u_lid 
        self.f = self.equilibrium(self.rho, self.u)

    def equilibrium(self, rho, u):
        # u: [2, ny, nx]
        # rho: [ny, nx]
        ux = u[0]
        uy = u[1]
        usq = ux**2 + uy**2
        
        feq = torch.zeros_like(self.f)
        for i in range(9):
            cu = self.cfg.cx[i] * ux + self.cfg.cy[i] * uy
            feq[i] = rho * self.cfg.w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
        return feq

    def collide_and_stream(self, force_field=None):
        # 1. 宏观量计算
        rho = torch.sum(self.f, dim=0)
        u = torch.zeros((2, self.ny, self.nx), device=DEVICE, dtype=DTYPE)
        u[0] = torch.sum(self.f * self.cfg.cx.view(9, 1, 1), dim=0) / rho
        u[1] = torch.sum(self.f * self.cfg.cy.view(9, 1, 1), dim=0) / rho
        
        self.rho = rho
        self.u = u

        # 2. 计算平衡态
        feq = self.equilibrium(rho, u)
        
        # 3. 外力项 (Guo Forcing)
        # g_K = (1 - 1/2tau) * w_k * [ (e-u)/cs^2 + (e.u)/cs^4 * e ] . F
        source_term = torch.zeros_like(self.f)
        if force_field is not None:
            # force_field shape: [2, ny, nx] (Fx, Fy)
            Fx = force_field[0]
            Fy = force_field[1]
            
            for i in range(9):
                ex_min_ux = self.cfg.cx[i] - u[0]
                ey_min_uy = self.cfg.cy[i] - u[1]
                
                e_dot_u = self.cfg.cx[i] * u[0] + self.cfg.cy[i] * u[1]
                e_dot_F = self.cfg.cx[i] * Fx + self.cfg.cy[i] * Fy
                
                # prefactor: (1 - 1/(2tau)) * w_i
                pref = (1.0 - 0.5 * self.cfg.omega) * self.cfg.w[i]
                
                term1 = (ex_min_ux * Fx + ey_min_uy * Fy) / self.cfg.cs_sq
                term2 = (e_dot_u * e_dot_F) / (self.cfg.cs_sq**2)
                
                source_term[i] = pref * (term1 + term2)

        # 4. 碰撞
        self.f = self.f - self.cfg.omega * (self.f - feq) + source_term

        # 5. 流动 (Streaming)
        for i in range(9):
            # 使用 torch.roll 实现周期性边界（模拟内部流动），物理边界条件在 apply_boundary_conditions 中修正
            self.f[i] = torch.roll(self.f[i], shifts=(int(self.cfg.cx[i]), int(self.cfg.cy[i])), dims=(1, 0))

        # 6. 边界条件
        self.apply_boundary_conditions()

    def apply_boundary_conditions(self):
        # 顶盖驱动流边界条件
        # y=0: Bottom Wall (No slip)
        # y=ny-1: Top Wall (Moving lid)
        # x=0: Left Wall (No slip)
        # x=nx-1: Right Wall (No slip)
        
        # 简化版 Bounce-back 处理
        # Left/Right/Bottom: Stationary
        self.f[[1, 5, 8], :, 0] = self.f[[3, 7, 6], :, 0]   # Left (x=0) reflects to Right
        self.f[[3, 6, 7], :, -1] = self.f[[1, 8, 5], :, -1] # Right (x=nx-1) reflects to Left
        self.f[[2, 5, 6], 0, :] = self.f[[4, 7, 8], 0, :]   # Bottom (y=0) reflects to Up
        
        # Top: Moving Lid (u_lid)
        u_w = self.cfg.u_lid
        # 近似动量修正反弹
        self.f[4, -1, :] = self.f[2, -1, :]
        # f7 (down-left) comes from f5 (up-right). 
        # f_new = f_old - 6 * w * rho * (c.u_w)
        # c7 = (-1, -1), u_w = (u, 0). c7.u_w = -u. -> + 6*w*rho*u
        self.f[7, -1, :] = self.f[5, -1, :] + 6 * self.cfg.w[7] * self.rho[-1, :] * u_w
        # f8 (down-right) comes from f6 (up-left).
        # c8 = (1, -1), u_w = (u, 0). c8.u_w = u. -> - 6*w*rho*u
        self.f[8, -1, :] = self.f[6, -1, :] - 6 * self.cfg.w[8] * self.rho[-1, :] * u_w

