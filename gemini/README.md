# LBM-NN-LES 2D Lid-Driven Cavity Flow

This project implements a hybrid Lattice Boltzmann Method (LBM) and Neural Network (NN) approach for Large Eddy Simulation (LES) of 2D turbulence.

## Structure

The code has been refactored into modular components:

1.  **`main.py`**: The entry point. Manages the simulation workflow:
    *   Runs Fine Grid Simulation (DNS).
    *   Filters data.
    *   Trains the SGS Neural Network.
    *   Runs Coarse Grid Simulation (LES) with the trained model.
    *   Visualizes results.
2.  **`LBM_Solver.py`**: Contains the `LBMSolver` and `LBMConfig` classes. Implements the D2Q9 LBM with Guo Forcing.
3.  **`SGS_Model.py`**: Defines the `SGSModel` neural network and input computation logic (`compute_nn_inputs`, `get_gradients`).
4.  **`train_utils.py`**: Contains helper functions for data filtering (`box_filter`) and model training (`train_sgs_model`).

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation:
```bash
python main.py
```

## Fixes & Improvements

- **OMP Error Fix**: Added `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` to `main.py` to resolve the OpenMP runtime conflict on macOS/Windows.
- **Convergence**: The neural network predicts the SGS stress tensor components directly from normalized flow features, avoiding the stability issues associated with log-transforms found in previous versions.
