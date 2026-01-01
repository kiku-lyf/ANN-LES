import torch
import torch.nn as nn
import torch.optim as optim
from SGS_Model import SGSModel, compute_nn_inputs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def box_filter(u, rho, coarse_nx, coarse_ny):
    # u: [2, ny, nx]
    # filter size n
    ny, nx = u.shape[1], u.shape[2]
    n = nx // coarse_nx
    
    # 简化版 Box Filter: 平均池化
    u_coarse = torch.nn.functional.avg_pool2d(u, kernel_size=n, stride=n)
    rho_coarse = torch.nn.functional.avg_pool2d(rho.unsqueeze(0), kernel_size=n, stride=n).squeeze(0)
    
    # 计算 SGS Stress tau_ij = mean(ui * uj) - mean(ui) * mean(uj)
    uu_fine = torch.zeros((3, ny, nx), device=DEVICE) # xx, xy, yy
    uu_fine[0] = u[0] * u[0]
    uu_fine[1] = u[0] * u[1]
    uu_fine[2] = u[1] * u[1]
    
    uu_coarse = torch.nn.functional.avg_pool2d(uu_fine, kernel_size=n, stride=n)
    
    u_mean_sq = torch.zeros_like(uu_coarse)
    u_mean_sq[0] = u_coarse[0] * u_coarse[0]
    u_mean_sq[1] = u_coarse[0] * u_coarse[1]
    u_mean_sq[2] = u_coarse[1] * u_coarse[1]
    
    tau = uu_coarse - u_mean_sq # [3, cny, cnx] -> xx, xy, yy
    
    return u_coarse, rho_coarse, tau

def train_sgs_model(u_data, tau_data, epochs=100):
    # u_data: list of u tensors from different time steps
    # tau_data: list of tau tensors
    
    model = SGSModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Preparing training data...")
    X_list = []
    Y_list = []
    
    for u, tau in zip(u_data, tau_data):
        # inputs
        inp = compute_nn_inputs(u) # [4, ny, nx]
        # permute to [ny*nx, 4]
        inp = inp.permute(1, 2, 0).reshape(-1, 4)
        
        # targets
        # tau: [3, ny, nx] (xx, xy, yy)
        # target outputs: tau_xx_a, tau_xy, tau_kk
        tau_xx = tau[0]
        tau_xy = tau[1]
        tau_yy = tau[2]
        tau_kk = tau_xx + tau_yy
        tau_xx_a = tau_xx - 0.5 * tau_kk
        
        target = torch.stack([tau_xx_a, tau_xy, tau_kk], dim=0) # [3, ny, nx]
        target = target.permute(1, 2, 0).reshape(-1, 3)
        
        X_list.append(inp)
        Y_list.append(target)
        
    X = torch.cat(X_list, dim=0).detach() # inputs
    Y = torch.cat(Y_list, dim=0).detach() # targets
    
    print(f"Training data shape: {X.shape}")
    
    print("Start Training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
    return model

