```markdown
# Fourier Neural Operator (FNO) for Pressure Field Prediction

## 1. Project Overview

This project implements a **Fourier Neural Operator (FNO)** for learning the mapping between permeability and pressure fields in a 2D porous medium.

The model learns a nonlinear operator:

G: a(x) → u(x)

where:
- a(x) — permeability field
- u(x) — pressure field

The goal is to approximate the solution operator of a PDE using data-driven learning.

---

## 2. Physical Problem Statement

The underlying physical model corresponds to the Darcy equation:

−∇ · (a(x) ∇u(x)) = f(x),   x ∈ Ω

where:
- Ω is a 2D spatial domain
- a(x) is spatial permeability
- u(x) is pressure

Important note:
- The PDE is NOT solved inside the neural network
- The dataset already contains numerical solutions
- The model learns the mapping (a(x) → u(x))

---

## 3. Project Structure

project/
│
├── fno_experiment.py          # training and experiments
├── check_data.py              # dataset analysis and visualization
├── utilities3.py              # data loading, normalization, losses
│
├── ...                        # datasets - нужно скачать
https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flu_lu_yale_edu%2FDocuments%2Fdatasets%2F2022_CMAME_Lu%2FDarcy_rectangular_PWC&viewid=e10cdfb7-ffdd-44de-b905-4560554a8b8f 
│
├── training_plots/            # training curves and results
├── dataset_visualization/     # dataset inspection plots
└── README.md

---

## 4. Dataset Description

Two datasets are used:

- piececonst_r421_N1024_smooth1.mat
- piececonst_r421_N1024_smooth2.mat

Each sample contains:

- coeff → permeability field a(x)
- sol → pressure field u(x)

Grid resolution:
- Original: 421 × 421
- Downsampled: 29 × 29

---

## 5. Data Analysis (check_data.py)

The script performs:

- loading MAT files
- checking tensor shapes
- comparing dataset statistics
- computing absolute differences
- visualizing fields and distributions

Outputs:

dataset_visualization/datasets_comparison.png

---

## 6. Model: Fourier Neural Operator (FNO)

The model is based on spectral convolution using FFT.

Main components:
- Fourier transform (FFT)
- learnable spectral weights
- inverse FFT
- residual connections (Conv1D)
- fully connected layers

---

## 7. Architecture

Input:
(a(x), spatial coordinates)

↓
Linear embedding (3 → width)

↓
FNO layers:
    FFT → spectral filtering → IFFT
    + residual branch

↓
Fully Connected layers

↓
Output:
u(x)

---

## 8. Frequency Analysis (FFT)

The project includes spectral analysis of pressure fields:

Operations:
- 2D FFT (fft2)
- frequency shift (fftshift)
- log magnitude spectrum

Visualized components:
- full frequency spectrum
- low-frequency structure
- high-frequency components

Outputs:

training_plots/*_fft.png

---

## 9. Training Procedure

Optimization setup:
- Optimizer: Adam
- Scheduler: StepLR
- Loss: MSE + L2 relative error

Tracked metrics:
- training MSE
- relative L2 error (train/test)

---

## 10. Training Curves

Saved in:

training_plots/<experiment_name>_loss.png

Includes:
- MSE loss (training)
- L2 error (test)

---

## 11. Experiments

The following configurations are compared:

- Baseline
- Layers_2
- Layers_6
- Width_64
- Modes_8

Purpose:
- analyze depth impact
- analyze width impact
- analyze spectral modes sensitivity

---

## 12. Prediction Visualization

For each experiment:

- input field (permeability + coordinates)
- ground truth pressure
- predicted pressure

Saved in:

training_plots/

---

## 13. Metrics

Main metrics:

### L2 Relative Error
Measures relative reconstruction error between fields.

### MSE
Pointwise reconstruction error.

---

## 14. utilities3.py

Contains:

- MatReader — dataset loader (MAT files)
- UnitGaussianNormalizer — normalization
- LpLoss — relative Lp error
- HsLoss — Sobolev-type loss
- DenseNet — fully connected baseline
- count_params — parameter counter

---

## 15. Key Idea

Instead of learning a function, the model learns an operator:

Gθ: a(x) → u(x)

This corresponds to learning the solution operator of the Darcy PDE:

−∇ · (a(x) ∇u(x)) = f(x)

---

## 16. Results

The project produces:

out/
- FFT analysis
- prediction visualization
- experiment comparison

dataset_visualization/
- dataset inspection plots

---

## 17. Running the Project

Install dependencies:
```

pip install numpy matplotlib torch scipy h5py

```

Check dataset:
```

python check_data.py

```

Train model:
```

python fno_experiment.py

```

