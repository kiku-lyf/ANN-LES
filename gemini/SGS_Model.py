import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGSModel(nn.Module):
    def __init__(self):
        super(SGSModel, self).__init__()
        # 输入: |omega|, theta, |A|, |S| (4 dim)
        # 输出: tau_xx_a, tau_xy, tau_kk (3 dim) -> reconstruct tau_xx, tau_xy, tau_yy
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # tau_xx_a, tau_xy, tau_kk
        )

    def forward(self, x):
        return self.net(x)

def get_gradients(f, dx=1.0):
    # f: [batch, h, w] or [h, w]
    # 使用中心差分计算梯度, 边界处理简化为单侧或保持内部
    if f.dim() == 2:
        f = f.unsqueeze(0)
    
    # Pad for central difference
    f_pad = torch.nn.functional.pad(f, (1, 1, 1, 1), mode='replicate')
    
    df_dx = (f_pad[:, 1:-1, 2:] - f_pad[:, 1:-1, :-2]) / (2 * dx)
    df_dy = (f_pad[:, 2:, 1:-1] - f_pad[:, :-2, 1:-1]) / (2 * dx) 
    
    return df_dx.squeeze(0), df_dy.squeeze(0)

def compute_nn_inputs(u_coarse):
    # u_coarse: [2, ny, nx]
    dx = 1.0 # coarse grid unit
    
    # Gradients
    dux_dx, dux_dy = get_gradients(u_coarse[0], dx)
    duy_dx, duy_dy = get_gradients(u_coarse[1], dx)
    
    # 1. Vorticity magnitude |omega| = |dv/dx - du/dy|
    omega = torch.abs(duy_dx - dux_dy)
    
    # 2. Divergence theta = du/dx + dv/dy
    theta = dux_dx + duy_dy
    
    # 3. Strain rate tensor S_ij
    # S_ij = 0.5 * (dui_dxj + duj_dxi)
    Sxx = dux_dx
    Syy = duy_dy
    Sxy = 0.5 * (dux_dy + duy_dx)
    
    # |S| = sqrt(2 * S_ij S_ij) ? PDF says sqrt(S_ij S_ij) or similar.
    # Usually |S| = sqrt(2 S_ij S_ij). 
    # But let's follow the PDF notation: sqrt(S_ij S_ij)
    # S_ij S_ij = Sxx^2 + Syy^2 + 2*Sxy^2
    S_mag = torch.sqrt(Sxx**2 + Syy**2 + 2 * Sxy**2)
    
    # 4. Velocity gradient tensor A_ij = di uj
    # A_ij A_ij = A11^2 + A12^2 + A21^2 + A22^2
    A_mag = torch.sqrt(dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2)
    
    # Stack inputs: [4, ny, nx]
    inputs = torch.stack([omega, theta, A_mag, S_mag], dim=0)
    return inputs

