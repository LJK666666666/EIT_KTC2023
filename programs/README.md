# EIT Programs Overview

This directory contains multiple Electrical Impedance Tomography (EIT) implementations, ranging from classical algorithms to state-of-the-art deep learning methods. Each program serves different purposes and uses different approaches to solve the EIT inverse problem.

## Quick Reference Table

| Program | Type | Language | Primary Use | Status |
|---------|------|----------|-------------|--------|
| [CDEIT](#1-cdeit) | Deep Learning | Python | Diffusion model reconstruction | Active |
| [eit_fenicsx](#2-eit_fenicsx) | FEM + ML Hybrid | Python | Complete electrode model solver | Active |
| [pyEIT](#3-pyeit) | Classical + Framework | Python | Modular EIT framework | Active |
| [eit-transformer](#4-eit-transformer) | Deep Learning | Python | Transformer-based reconstruction | Research |
| [Weighted-SBL](#5-weighted-sparse-bayesian-learning) | Bayesian | MATLAB/Python | Sparse Bayesian learning | Research |
| [OpenEIT](#6-openeit) | Hardware + Software | Python | Real-time EIT dashboard | Active |
| [awesome-eit](#7-awesome-eit) | Resource List | - | Curated EIT resources | Reference |
| [ktc2023_postprocessing](#8-ktc2023_postprocessing) | Deep Learning (KTC2023) | Python | Postprocessing UNet for KTC2023 | Research |
| [ktc2023_fcunet](#9-ktc2023_fcunet) | Deep Learning (KTC2023) | Python | FC-UNet for KTC2023 | Research |
| [ktc2023_conditional_diffusion](#10-ktc2023_conditional_diffusion) | Deep Learning (KTC2023) | Python | Conditional diffusion for KTC2023 | Research |

---

## 1. CDEIT

**Conditional Diffusion Model for Electrical Impedance Tomography**

### Overview
State-of-the-art deep learning approach using conditional diffusion models (DiT - Diffusion Transformer) for EIT image reconstruction. Achieves high-quality reconstructions by treating EIT as an image generation task conditioned on voltage measurements.

### Key Features
- **Diffusion-based reconstruction**: Leverages denoising diffusion probabilistic models
- **Multi-dataset support**: Works with simulated data, UEF2017, and KTC2023
- **GPU acceleration**: Multi-GPU training with `accelerate`
- **Gauss-Newton conditioning**: Uses traditional methods as conditioning signals

### Quick Start
```bash
cd CDEIT

# Training
python main.py --mode train

# Testing on KTC2023
python main.py --mode test --data ktc2023
```

### Requirements
- PyTorch 2.3.1
- accelerate, ema_pytorch
- CUDA-capable GPU recommended

### Publication
Shi, S., Kang, R., & Liatsis, P. (2025). A Conditional Diffusion Model for Electrical Impedance Tomography Image Reconstruction. *IEEE Transactions on Instrumentation and Measurement*.

---

## 2. eit_fenicsx

**FEM Complete Electrode Model Solver**

### Overview
Professional-grade finite element method (FEM) solver implementing the Complete Electrode Model (CEM) using FEniCSx. Supports both classical reconstruction algorithms and modern deep learning approaches.

### Key Features
- **Complete Electrode Model**: Accurate modeling of contact impedance
- **Multiple backends**: PETSc (fast) or Scipy (portable)
- **Rich algorithm suite**:
  - Gauss-Newton with various regularizers (Tikhonov, TV, L1)
  - Data-driven reconstructors (UNet, Attention UNet, FNO)
  - Learned iterative reconstruction
- **Jacobian computation**: Efficient adjoint-based Jacobian calculation
- **PyTorch integration**: Differentiable forward solver for learning

### Quick Start
```bash
cd eit_fenicsx

# Install FEniCSx
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista

# Run example
python examples/gauss_newton_tv.py
```

### Architecture Highlights
- Forward solver: Solves `-∇·(σ∇u) = 0` with contact impedance BCs
- Reconstruction: Iterative methods or neural network-based
- Meshing: Supports Gmsh meshes with arbitrary electrode configurations

### Publication
Denker, A., Kereta, Ž., Singh, I., Freudenberg, T., Kluth, T., Maass, P., & Arridge, S. (2024). Data-driven approaches for electrical impedance tomography image segmentation from partial boundary data. *Applied Mathematics for Modern Challenges*.

---

## 3. pyEIT

**Python EIT Framework**

### Overview
The most widely-used open-source EIT framework. Designed with modularity, minimalism, and extensibility in mind. Excellent for teaching, prototyping, and research.

### Key Features
- **Multiple algorithms**:
  - Jacobian-based (JAC)
  - Back-projection (BP)
  - GREIT (Graz consensus)
  - SVD-based methods
- **2D and 3D support**: Full 3D reconstruction capabilities
- **Mesh generation**: Built-in distmesh integration
- **Visualization**: 2D/3D plotting with matplotlib and vispy
- **Well-documented**: Extensive examples and tutorials

### Installation
```bash
# Via pip (recommended)
pip install pyeit

# Via conda
conda install -c conda-forge pyeit
```

### Quick Start
```bash
cd pyEIT

# 2D Jacobian reconstruction
python examples/eit_dynamic_jac.py

# GREIT reconstruction
python examples/eit_dynamic_greit.py

# 3D forward simulation
python examples/fem_forward3d.py
```

### Use Cases
- Educational purposes (learning EIT basics)
- Algorithm prototyping
- Comparison baseline for new methods
- Integration into larger systems

### Publication
Liu, B., Yang, B., Xu, C., Xia, J., Dai, M., Ji, Z., You, F., Dong, X., Shi, X., & Fu, F. (2018). pyEIT: A python based framework for Electrical Impedance Tomography. *SoftwareX*, 7, 304-308.

---

## 4. eit-transformer

**Transformer Meets Boundary Value Inverse Problems**

### Overview
Novel approach applying transformer architecture to EIT reconstruction. Uses modified 2D attention operators based on integral operator formulation, with PDE-based feature preprocessing.

### Key Features
- **U-Integral Transformer (UIT)**: Modified attention mechanism for integral operators
- **Multiple variants**:
  - `uit`: Integral transformer (base)
  - `ut`: Traditional softmax normalization
  - `hut`: Hybrid with linear attention
  - `xut`: Cross-attention with Hadamard product
  - `fno2d`: Fourier neural operator baseline
- **PDE-based features**: Preprocesses boundary measurements using PDE operators
- **Few-shot learning**: Can work with single or few current-voltage pairs

### Quick Start
```bash
cd eit-transformer

# Train U-integral transformer
python run_train.py --model uit --parts 2 4 5 6

# Evaluate
python evaluation.py --model uit
```

### Research Focus
Exploring attention mechanisms for inverse problems, particularly suitable for:
- Sparse measurement scenarios
- Multi-physics inverse problems
- Transfer learning across different geometries

### Publication
Guo, R., Cao, S., & Chen, L. (2022). Transformer Meets Boundary Value Inverse Problems. *arXiv preprint arXiv:2209.14977*.

---

## 5. Weighted-Sparse-Bayesian-Learning

**Weighted Bound-Optimization Block Sparse Bayesian Learning**

### Overview
Advanced Bayesian approach for EIT reconstruction using weighted sparse priors. Implements bound-optimization for efficient sparse Bayesian learning with superior performance on noisy measurements.

### Key Features
- **Sparse Bayesian Learning (SBL)**: Automatic relevance determination
- **Weighted block structure**: Exploits spatial correlations
- **Bound-optimization**: Efficient EM-like algorithm
- **Comparison framework**: Benchmarks against L1, L2, and standard SBL
- **MATLAB + Python**: MATLAB main implementation with Python demo

### Quick Start
```bash
cd Weighted-Sparse-Bayesian-Learning-for-Electrical-Impedance-Tomography

# MATLAB version
matlab -r "run Main.m"

# Python demo (under development)
python python_demo.py
```

### Dependencies
- **MATLAB**: Requires EIDORS 3.9+
- **Python**: NumPy, SciPy, Matplotlib

### Data Support
- KIT4 experimental data (from Zenodo)
- EIDORS in-vivo thoracic data
- Custom 2D geometries

### Publication
Dimas, C., Alimisis, V., & Sotiriadis, P. P. (2022). Electrical Impedance Tomography using a Weighted Bound-Optimization Block Sparse Bayesian Learning Approach. *IEEE BIBE 2022*.

---

## 6. OpenEIT

**Open Source EIT Hardware and Software Platform**

### Overview
Complete EIT solution combining hardware design (SPECTRA board) with real-time Python dashboard. Aimed at making biomedical imaging accessible, hackable, and educational.

### Key Features
- **Hardware integration**: Works with SPECTRA EIT device (2" square PCB)
- **Real-time reconstruction**: Live imaging with multiple algorithms
- **Multiple modalities**:
  - Time series bioimpedance
  - Bioimpedance spectroscopy
  - Electrical impedance tomography
- **Data recording**: Save and replay measurements
- **Bluetooth + Serial**: Wireless or wired communication
- **Web-based dashboard**: Browser-accessible interface

### Installation
```bash
cd OpenEIT

# Create virtual environment (recommended)
virtualenv -p python3 envtest
source envtest/bin/activate  # On Windows: envtest\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python app.py
```

Access at: http://127.0.0.1:8050/

### Algorithms Implemented
- Back Projection (BP)
- Graz Consensus (GREIT)
- Gauss-Newton
- All powered by pyEIT backend

### Hardware
- **SPECTRA board**: Open-source PCB design available
- **32 electrodes**: Configurable patterns
- **Multi-frequency**: Up to 250 kHz
- **Portable**: Battery-powered with Bluetooth

### Use Cases
- Educational demonstrations
- DIY biomedical imaging
- Lung imaging (ventilation monitoring)
- Research prototyping

### License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0

---

## 7. awesome-eit

**Curated List of EIT Resources**

### Overview
Comprehensive collection of EIT-related libraries, tools, hardware, algorithms, and applications. Community-maintained repository for discovering EIT resources.

### Categories
- **Simulation libraries**: pyEIT, EIDORS, CEIT, MIT forward solvers
- **Hardware**: ScouseTom, OpenEIT, EIT-kit
- **Algorithms**: pydbar, DeepDbar, various reconstruction methods
- **Meshing**: eitmesh, Mesher, tetrahedralizer
- **Visualization**: Datoviz, vedo, polyscope
- **Applications**: Clinical datasets, electrode selection, ML methods
- **Community**: Related awesome lists and resources

### How to Use
Browse the repository to discover:
- New EIT software to try
- Hardware designs for DIY projects
- Datasets for benchmarking
- Related communities and conferences

### Contributing
Fork the repository and submit a PR with new resources!

---

## Choosing the Right Program

### For Beginners
- **pyEIT**: Best starting point, excellent documentation
- **OpenEIT**: If you have hardware and want hands-on experience

### For Research (Classical Methods)
- **eit_fenicsx**: Most accurate forward model, professional FEM solver
- **Weighted-SBL**: State-of-the-art Bayesian reconstruction
- **pyEIT**: Quick prototyping and comparisons

### For Research (Deep Learning)
- **CDEIT**: Best reconstruction quality, diffusion models
- **eit-transformer**: Novel architectures, attention mechanisms
- **eit_fenicsx**: Hybrid model-based + data-driven approaches

### For KTC2023 Challenge
- **ktc2023_fcunet**: Best scores (0.985-0.990), but finetuned on challenge data
- **ktc2023_postprocessing**: Good scores (0.643-0.881), generalizes better
- **ktc2023_conditional_diffusion**: Probabilistic with uncertainty quantification

### For Real-Time Applications
- **OpenEIT**: Hardware integration, live imaging
- **pyEIT**: Fast algorithms (BP, GREIT)

### For Benchmarking
Use multiple programs:
1. **pyEIT** for classical baselines (BP, GREIT, Gauss-Newton)
2. **CDEIT** for state-of-the-art deep learning
3. **eit_fenicsx** for accurate forward modeling
4. **ktc2023_fcunet** for reference upper bound (if using KTC2023 data)

---

## Common Datasets

Several programs can work with the same datasets:

### KTC2023 (Kuopio Tomography Challenge)
- **Supported by**: CDEIT, eit_fenicsx, ktc2023_postprocessing, ktc2023_fcunet, ktc2023_conditional_diffusion
- **Location**: `../data/KTC2023/` or download from Zenodo
- **Features**: Multiple difficulty levels (1-7), missing electrodes, segmentation task

### UEF2017 (2D Experimental Data)
- **Supported by**: CDEIT
- **Source**: Finnish Inverse Problems Society
- **Features**: Real water tank measurements

### KIT4 (Experimental Data)
- **Supported by**: eit_fenicsx, Weighted-SBL
- **Source**: Zenodo (1203914)
- **Features**: 16 electrode circular phantom

### Simulated Data
All programs can generate their own simulated data using:
- EIDORS (MATLAB)
- pyEIT mesh generators
- FEniCSx forward solvers

---

## Execution Paths

**IMPORTANT**: Different programs have different working directory requirements.

| Program | Execute From | Example |
|---------|--------------|---------|
| CDEIT | `programs/CDEIT/` | `cd programs/CDEIT && python main.py` |
| eit_fenicsx | `programs/eit_fenicsx/` | `cd programs/eit_fenicsx && python examples/gauss_newton_tv.py` |
| pyEIT | `programs/pyEIT/` | `cd programs/pyEIT && python examples/eit_dynamic_jac.py` |
| eit-transformer | `programs/eit-transformer/` | `cd programs/eit-transformer && python run_train.py` |
| Weighted-SBL | `programs/Weighted-Sparse-Bayesian-Learning-for-Electrical-Impedance-Tomography/` | MATLAB or Python |
| OpenEIT | `programs/OpenEIT/` | `cd programs/OpenEIT && python app.py` |
| ktc2023_postprocessing | `programs/ktc2023_postprocessing/` | `cd programs/ktc2023_postprocessing && python main.py` |
| ktc2023_fcunet | `programs/ktc2023_fcunet/` | `cd programs/ktc2023_fcunet && python main.py` |
| ktc2023_conditional_diffusion | `programs/ktc2023_conditional_diffusion/` | `cd programs/ktc2023_conditional_diffusion && python main.py` |

All programs use relative imports and relative paths, requiring execution from their project directories.

---

## Dependencies Overview

### Python-based Programs

**Common dependencies:**
- NumPy, SciPy, Matplotlib
- For deep learning: PyTorch (or TensorFlow)

**Specialized:**
- **eit_fenicsx**: FEniCSx, PETSc, MPI
- **CDEIT**: accelerate, ema_pytorch, timm
- **pyEIT**: vispy (3D viz), shapely (thorax mesh)
- **OpenEIT**: Dash, plotly (web dashboard)

### MATLAB Programs
- **Weighted-SBL**: EIDORS 3.9+

---

## Contributing

To contribute to any of these projects:

1. **pyEIT**: Upstream at [eitcom/pyEIT](https://github.com/eitcom/pyEIT)
2. **CDEIT**: Contact shuaikai.shi@ku.ac.ae
3. **eit_fenicsx**: Research collaboration with UCL/Bremen teams
4. **OpenEIT**: GitHub issues and PRs welcome
5. **awesome-eit**: Fork and submit PRs

---

## Additional Resources

### Documentation
- **pyEIT**: Most comprehensive documentation and examples
- **eit_fenicsx**: See README and example scripts
- **OpenEIT**: Tutorials at [readthedocs](https://openeitgithubio.readthedocs.io)

### Related Software
- **EIDORS** (MATLAB): The original EIT software, still widely used
- **GREIT** (MATLAB): Graz consensus reconstruction
- **KTC Challenge**: Benchmark datasets and scoring

### Community
- Join mailing lists through project websites
- Check **awesome-eit** for conferences and workshops
- Stack Overflow tag: `electrical-impedance-tomography`

---

## Citation

When using these programs in research, please cite the appropriate papers (see individual sections above). General EIT review:

```bibtex
@article{holder2005electrical,
  title={Electrical impedance tomography: methods, history and applications},
  author={Holder, David S},
  journal={Institute of Physics Publishing},
  year={2005}
}
```

---

## Maintenance Status

| Program | Last Updated | Active Development |
|---------|--------------|-------------------|
| pyEIT | Ongoing | ✅ Active (community) |
| CDEIT | 2025 | ✅ Active (research) |
| eit_fenicsx | 2024 | ✅ Active (research) |
| OpenEIT | Ongoing | ✅ Active (community) |
| eit-transformer | 2022 | ⚠️ Research code |
| Weighted-SBL | 2022 | ⚠️ Research code |
| awesome-eit | Ongoing | ✅ Active (curated list) |
| ktc2023_postprocessing | 2023 | ⚠️ KTC2023 submission |
| ktc2023_fcunet | 2023 | ⚠️ KTC2023 submission |
| ktc2023_conditional_diffusion | 2023 | ⚠️ KTC2023 submission |

---

## 8. ktc2023_postprocessing

**Postprocessing UNet for KTC2023 Challenge**

### Overview
KTC2023 challenge submission using a postprocessing UNet for EIT segmentation. The network takes linearized reconstructions as input and outputs segmentation masks. Trained on ~15,000 synthetic phantoms per difficulty level.

### Key Features
- **Postprocessing approach**: Refines classical reconstructions using deep learning
- **Multiple regularization inputs**: Uses 5 different regularization combinations (smoothness, Laplace, NOSER)
- **Level-aware**: Adapts to different difficulty levels (missing electrodes)
- **Jacobian-based preprocessing**: Uses precomputed Jacobian matrices (no FEniCS needed)
- **256×256 output**: Direct pixel-grid segmentation

### Method
```
Input: 5 regularized reconstructions (5×256×256) + difficulty level
        ↓
    Postprocessing UNet
        ↓
Output: Segmentation mask (256×256)
```

### Quick Start
```bash
cd ktc2023_postprocessing

# Create environment
conda env create -f environment.yml
conda activate ktc2023_postprocessing

# Download weights from:
# https://seafile.zfn.uni-bremen.de/d/faaf3799e6e247198a23/
# Place in: postprocessing_model/version_01/model.pt

# Download precomputed Jacobian/matrices from:
# https://seafile.zfn.uni-bremen.de/d/9108bc95b2e84cd285f8/
# Place in: data/

# Reconstruct
python main.py /path_to_input /path_to_output difficulty_level
```

### Performance (KTC2023 Score)

| Level | Score | Description |
|-------|-------|-------------|
| 1 | 0.873 | Full 32 electrodes |
| 2 | 0.881 | 1 electrode missing |
| 3 | 0.866 | 2 electrodes missing |
| 4 | 0.838 | 3 electrodes missing |
| 5 | 0.791 | 4 electrodes missing |
| 6 | 0.643 | 5 electrodes missing |
| 7 | 0.681 | 6 electrodes missing |

### Training Data
- **Synthetic phantoms**: Circles, polygons, hand-drawn objects (1-4 per image)
- **Forward operator**: Provided by KTC2023 organizers (dense mesh)
- **Reconstruction mesh**: Custom mesh for Jacobian computation

### Authors
Alexander Denker¹, Tom Freudenberg¹, Željko Kereta², Imraj RD Singh², Tobias Kluth¹, Peter Maass¹, Simon Arridge²

¹University of Bremen, ²University College London

---

## 9. ktc2023_fcunet

**FC-UNet for KTC2023 Challenge**

### Overview
KTC2023 challenge submission using a Fully Connected UNet (FC-UNet) that directly processes measurement differences. Achieves near-perfect scores after finetuning on challenge data. Architecture inspired by [Z. Chen et al. IEEE TMI 2020](https://ieeexplore.ieee.org/document/9128764).

### Key Features
- **End-to-end learning**: Direct measurements → segmentation (no intermediate reconstruction)
- **FC-UNet architecture**: Linear layer (2356→64×64) + bilinear upsampling (→256×256) + UNet
- **Three-phase training**:
  1. Train linear layer only (synthetic data)
  2. Train full network (synthetic data)
  3. Finetune on challenge phantoms (real measurements)
- **Level-aware**: Difficulty level as additional input
- **Missing measurement handling**: Fills with zeros

### Method
```
Input: Voltage difference (U - Uref) ∈ ℝ²³⁵⁶ + difficulty level
        ↓
    Linear layer → 64×64
        ↓
    Bilinear upsample → 256×256
        ↓
    UNet (segmentation)
        ↓
Output: Segmentation mask (256×256)
```

### Quick Start
```bash
cd ktc2023_fcunet

# Create environment
conda env create -f environment.yml
conda activate ktc2023_fcunet

# Download weights from:
# https://seafile.zfn.uni-bremen.de/d/f22fb47500f34b0fb5c2/
# Place in: fcunet_model/

# Reconstruct
python main.py /path_to_input /path_to_output difficulty_level
```

### Performance (KTC2023 Score)

| Level | Score | Description |
|-------|-------|-------------|
| 1 | **0.988** | Full 32 electrodes |
| 2 | **0.990** | 1 electrode missing |
| 3 | **0.989** | 2 electrodes missing |
| 4 | **0.989** | 3 electrodes missing |
| 5 | **0.989** | 4 electrodes missing |
| 6 | **0.986** | 5 electrodes missing |
| 7 | **0.985** | 6 electrodes missing |

**Note**: These results may be overly optimistic due to finetuning on the challenge data (potential overfitting).

### Architecture Highlights
- **Input dimension**: 2356 measurements (76 patterns × 31 measurements/pattern)
- **First layer**: Linear transformation to spatial domain
- **UNet backbone**: Standard encoder-decoder with skip connections
- **Output**: Binary segmentation (conductive/resistive)

### Training Strategy
1. **Phase 1 (Linear layer)**: Learn measurement → spatial mapping
2. **Phase 2 (Full UNet)**: Learn segmentation from spatial features
3. **Phase 3 (Finetuning)**: Adapt to real hardware characteristics

### Authors
Alexander Denker¹, Tom Freudenberg¹, Željko Kereta², Imraj RD Singh², Tobias Kluth¹, Peter Maass¹, Simon Arridge²

¹University of Bremen, ²University College London

---

## 10. ktc2023_conditional_diffusion

**Conditional Diffusion Model for KTC2023 Challenge**

### Overview
KTC2023 challenge submission using conditional diffusion models for probabilistic EIT segmentation. Trains separate diffusion models for each difficulty level. Samples multiple reconstructions and takes pixel-wise majority vote for final prediction.

### Key Features
- **Probabilistic reconstruction**: Generates multiple plausible segmentations
- **Uncertainty quantification**: Can provide pixel-wise uncertainty maps
- **DDIM sampling**: Fast deterministic sampling (fewer steps than DDPM)
- **Per-level models**: 7 separate diffusion models (one per difficulty level)
- **Ensemble prediction**: Takes N random samples and votes
- **Guided diffusion backbone**: Based on OpenAI's guided-diffusion implementation

### Method
```
Input: 5 regularized reconstructions (5×256×256) + difficulty level
        ↓
    Conditional Diffusion Model (per level)
        ↓
    Sample N times (parallel or sequential)
        ↓
    Majority vote per pixel
        ↓
Output: Segmentation mask (256×256)
```

### Diffusion Process
**Forward (training)**: `σₜ = √αₜ σ₀ + √(1-αₜ) εₜ` (add noise)
**Reverse (inference)**: Start from noise, iteratively denoise using `ε_θ(σₜ, c, t)`

### Quick Start
```bash
cd ktc2023_conditional_diffusion

# Create environment
conda env create -f environment.yml
conda activate ktc2023_conditional_diffusion

# Download weights (7 models) from:
# https://seafile.zfn.uni-bremen.de/d/59c291e4bf7d4064a1be/
# Place in: diffusion_models/level_{1..7}/model.pt

# Download precomputed Jacobian/matrices from:
# https://seafile.zfn.uni-bremen.de/d/9108bc95b2e84cd285f8/
# Place in: data/

# Reconstruct
python main.py /path_to_input /path_to_output difficulty_level

# Batch mode control (line 14 in main.py):
# BATCH_MODE = True   # Parallel sampling (faster, needs GPU)
# BATCH_MODE = False  # Sequential sampling (slower, less memory)
```

### Performance (KTC2023 Score)

| Level | Score | Description |
|-------|-------|-------------|
| 1 | 0.86 | Full 32 electrodes |
| 2 | 0.84 | 1 electrode missing |
| 3 | 0.83 | 2 electrodes missing |
| 4 | 0.83 | 3 electrodes missing |
| 5 | 0.82 | 4 electrodes missing |
| 6 | 0.72 | 5 electrodes missing |
| 7 | 0.79 | 6 electrodes missing |

### Advantages
- **Probabilistic**: Can quantify uncertainty
- **Flexible**: Can generate diverse plausible solutions
- **Noise robust**: Denoising process helps with measurement noise

### Disadvantages
- **Slow**: Requires multiple sampling steps (even with DDIM)
- **Memory intensive**: Needs to store/run multiple samples
- **Complex training**: Requires careful tuning of diffusion parameters

### Comparison with ktc2023_postprocessing
- **Same input**: Both use 5 regularized reconstructions
- **Same architecture**: Both use similar UNet backbones
- **Different paradigm**: Postprocessing (deterministic) vs Diffusion (probabilistic)
- **Speed**: Postprocessing is faster (single forward pass)
- **Quality**: Postprocessing slightly better for this challenge

### Authors
Alexander Denker¹, Tom Freudenberg¹, Željko Kereta², Imraj RD Singh², Tobias Kluth¹, Peter Maass¹, Simon Arridge²

¹University of Bremen, ²University College London

---

*For detailed usage of each program, refer to their individual README files in their respective directories.*
