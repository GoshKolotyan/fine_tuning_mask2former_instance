# Detectron2 Environment Setup

This repository contains an **environment.yml** file, which defines a conda environment for:
- **Python 3.10**  
- **PyTorch 1.11** (CUDA 11.3)  
- **Detectron2 0.6** (compiled for PyTorch 1.11 + CUDA 11.3)  
- **Numpy 1.23.x** (to avoid NumPy 2.x incompatibility)  
- **OpenCV** (for `import cv2`)

## Prerequisites

1. **Conda / Miniconda / Anaconda**  
   Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com) installed.
2. **GPU & CUDA Drivers**  
   Ensure your machine has a compatible **GPU driver** for CUDA 11.3. Typically, you need driver version >= 450.80.

## 1. Creating the Environment

To create the environment from the provided `environment.yml` file, run:

```bash
conda env create -f environment.yml
```

This command will:
- Create a conda environment named `detectron2_env` (as specified in `environment.yml`).
- Install Python 3.10, PyTorch 1.11 (with CUDA 11.3), Detectron2 0.6, Numpy 1.23, OpenCV, etc.

### Activating the Environment

Once created, activate it using:

```bash
conda activate detectron2_env
```

## 2. Verifying the Installation

After activation, verify the key packages:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import detectron2; print('Detectron2 version:', detectron2.__version__)"
python -c "import cv2; print('OpenCV import successful!')"
python -c "import torch; print('CUDA available?', torch.cuda.is_available())"
```

You should see:
- **PyTorch version:** `1.11.0+cu113` (or similar)  
- **Detectron2 version:** `0.6`  
- **OpenCV import successful!**  
- **CUDA available?:** `True` (if you have a GPU with the correct driver)

## 3. Using the Environment for Development

Whenever you want to run scripts that depend on Detectron2 or PyTorch in this environment, make sure you have activated it:

```bash
conda activate detectron2_env
python your_script.py
```

If you plan to use **Jupyter notebooks**:
```bash
conda install -c conda-forge jupyterlab
jupyter lab
```
Make sure the kernel is set to **detectron2_env** in Jupyter.

## 4. Troubleshooting

1. **NumPy 2.x Issue**  
   PyTorch 1.11 and older compiled C++/CUDA extensions may fail if you upgrade to NumPy >= 2.0. Hence, this environment pins NumPy to a 1.x version (e.g., 1.23.5). If you see error messages about “compiled using NumPy 1.x” vs. “NumPy 2.2.x,” re-install or downgrade NumPy:
   ```bash
   pip uninstall numpy
   pip install numpy==1.23.5
   ```

2. **Driver / Toolkit Mismatch**  
   Ensure your GPU driver is compatible with CUDA 11.3. You can verify with `nvidia-smi` and `nvcc --version`.

3. **Import Errors**  
   - If you see “No module named x,” check that you’re in the correct conda environment (`conda activate detectron2_env`).  
   - Run `conda list` to confirm the package is installed inside the environment.

## 5. Updating the Environment

To update packages or add new ones, you can modify `environment.yml` and re-run:
```bash
conda env update -f environment.yml
```
Or install packages directly with `conda install <pkg_name>` or `pip install <pkg_name>` inside the environment.

