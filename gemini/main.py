import os

# 解决 OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from LBM_Solver import LBMConfig, LBMSolver
from SGS_Model import SGSModel, get_gradients, compute_nn_inputs
from train_utils import box_filter, train_sgs_model

# 文件路径配置
FINE_DATA_PATH = "fine_data.pt"
MODEL_PATH = "sgs_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_fine_simulation(nx=128, ny=128, max_steps=100000, tolerance=1e-5):
    """
    Run fine grid simulation until convergence or max_steps.
    """
    # 检查是否已有数据
    if os.path.exists(FINE_DATA_PATH):
        print(f"Loading existing fine grid data from {FINE_DATA_PATH}...")
        data = torch.load(FINE_DATA_PATH, map_location=DEVICE)
        return data['u'], data['rho']

    print(f"Starting Fine Simulation {nx}x{ny}...")
    cfg = LBMConfig(nx, ny, Re=1000.0, u_lid=0.1)
    solver = LBMSolver(cfg)

    print(f"Target convergence error: {tolerance}")

    for t in range(max_steps):
        # Save previous state for error calculation
        if t % 100 == 0:
            u_old = solver.u.clone()

        solver.collide_and_stream()

        if t % 100 == 0 and t > 0:
            # Calculate relative L2 error
            diff = torch.norm(solver.u - u_old)
            norm = torch.norm(solver.u)
            error = diff / (norm + 1e-6)

            if t % 1000 == 0:
                print(f"Fine Step {t}, Error: {error.item():.2e}")

            if error < tolerance:
                print(f"Converged at step {t} with error {error.item():.2e}")
                break
    else:
        print(f"Warning: Did not converge within {max_steps} steps. Final error: {error.item():.2e}")

    # 保存数据
    print(f"Saving fine grid data to {FINE_DATA_PATH}...")
    torch.save({'u': solver.u, 'rho': solver.rho}, FINE_DATA_PATH)

    return solver.u, solver.rho


def run_les_simulation(model, nx=32, ny=32, steps=2000):
    print(f"Starting LES Simulation {nx}x{ny}...")
    cfg = LBMConfig(nx, ny, Re=1000.0, u_lid=0.1)  # Match Re
    solver = LBMSolver(cfg)

    for t in range(steps):
        # 1. Compute inputs for NN
        with torch.no_grad():
            inp_tensor = compute_nn_inputs(solver.u)  # [4, ny, nx]
            # [batch, 4] for NN
            inp_flat = inp_tensor.permute(1, 2, 0).reshape(-1, 4)

            # 2. Predict tau
            pred = model(inp_flat)  # [ny*nx, 3]
            pred = pred.reshape(ny, nx, 3).permute(2, 0, 1)  # [3, ny, nx]

            # Reconstruct tau tensor
            tau_xx_a = pred[0]
            tau_xy = pred[1]
            tau_kk = pred[2]

            tau_xx = tau_xx_a + 0.5 * tau_kk
            tau_yy = 0.5 * tau_kk - tau_xx_a  # since tau_yy_a = -tau_xx_a

            # 3. Compute Force F = div(tau)
            # Fx = d(tau_xx)/dx + d(tau_xy)/dy
            # Fy = d(tau_xy)/dx + d(tau_yy)/dy

            dtxx_dx, dtxx_dy = get_gradients(tau_xx)
            dtxy_dx, dtxy_dy = get_gradients(tau_xy)
            dtyy_dx, dtyy_dy = get_gradients(tau_yy)

            Fx = dtxx_dx + dtxy_dy
            Fy = dtxy_dx + dtyy_dy

            force = torch.stack([Fx, Fy], dim=0)

        # 4. Step LBM with Force
        solver.collide_and_stream(force_field=force)

        if t % 100 == 0:
            print(f"LES Step {t}/{steps}")

    return solver.u


def get_or_train_model(u_coarse_gt, tau_gt, epochs=1000):
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = SGSModel().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model

    print("Training NN...")
    # Increase epochs to ensure sufficient training
    model = train_sgs_model([u_coarse_gt], [tau_gt], epochs=epochs)

    print(f"Saving model to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    return model


def main():
    # 1. Fine Simulation (Generate Data or Load)
    FINE_NX = 1024
    COARSE_NX = 128

    # Run until convergence (error < 1e-5) or load from file
    u_fine, rho_fine = run_fine_simulation(nx=FINE_NX, ny=FINE_NX, max_steps=50000, tolerance=1e-5)

    # 2. Filter & Prepare Data
    print("Filtering data...")
    u_coarse_gt, rho_coarse_gt, tau_gt = box_filter(u_fine, rho_fine, COARSE_NX, COARSE_NX)

    # 3. Train NN or Load Model
    model = get_or_train_model(u_coarse_gt, tau_gt, epochs=10000)

    # 4. Run LES
    print("Running LES...")
    u_les = run_les_simulation(model, nx=COARSE_NX, ny=COARSE_NX, steps=10000)
#
#     # 5. Visualization
#     print("Visualizing...")
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 3, 1)
#     plt.title("Fine Grid U_mag (Filtered)")
#     umag_gt = torch.sqrt(u_coarse_gt[0] ** 2 + u_coarse_gt[1] ** 2).cpu()
#     plt.imshow(umag_gt, origin='lower')
#     plt.colorbar()
#
#     plt.subplot(1, 3, 2)
#     plt.title("LES U_mag")
#     umag_les = torch.sqrt(u_les[0] ** 2 + u_les[1] ** 2).cpu()
#     plt.imshow(umag_les, origin='lower')
#     plt.colorbar()
#
#     plt.subplot(1, 3, 3)
#     plt.title("Difference")
#     plt.imshow(torch.abs(umag_gt - umag_les), origin='lower')
#     plt.colorbar()
#
#     plt.savefig('lbm_nn_les_result.png')
#     print("Done. Result saved to lbm_nn_les_result.png")
#
#
# if __name__ == "__main__":
#     main()
    # 5. Visualization & Error Analysis
    print("Visualizing...")
    plt.figure(figsize=(15, 5))

    # Calculate magnitudes
    umag_gt = torch.sqrt(u_coarse_gt[0] ** 2 + u_coarse_gt[1] ** 2).cpu()
    umag_les = torch.sqrt(u_les[0] ** 2 + u_les[1] ** 2).cpu()

    # Calculate errors
    diff = umag_gt - umag_les
    abs_diff = torch.abs(diff)

    # L2 Error
    l2_error = torch.norm(diff) / torch.norm(umag_gt)
    # Max Error
    max_error = torch.max(abs_diff)
    # Mean Error
    mean_error = torch.mean(abs_diff)

    print(f"\nError Analysis:")
    print(f"Relative L2 Error: {l2_error.item():.6e}")
    print(f"Max Absolute Error: {max_error.item():.6e}")
    print(f"Mean Absolute Error: {mean_error.item():.6e}")

    plt.subplot(1, 3, 1)
    plt.title("Fine Grid U_mag (Filtered)")
    plt.imshow(umag_gt, origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("LES U_mag")
    plt.imshow(umag_les, origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title(f"Difference\nL2 Error: {l2_error.item():.2e}")
    plt.imshow(abs_diff, origin='lower')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('lbm_nn_les_result.png')
    print("Done. Result saved to lbm_nn_les_result.png")


if __name__ == "__main__":
    main()
