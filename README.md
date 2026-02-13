# Spiking Neural Networks for Ultra Low Power Event-Based Optical Flow Estimation

This repository implements Spiking Neural Networks (SNNs) for event-based optical flow estimation, based on the FireNet architecture. All models use LIF (Leaky Integrate-and-Fire) neurons for ultra-low-power inference, with optional quantization support for hardware deployment.

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)
- [Quick Start](#-quick-start)
- [Main Scripts](#-main-scripts)
- [Auxiliary Scripts](#-auxiliary-scripts)
- [Model Architectures](#-model-architectures)
- [Configuration](#-configuration)
- [ONNX Export](#-onnx-export)
- [Citation](#-citation)

---

## 📁 Repository Structure

```
📦Event_Flow
 ┣ 📂configs/              Configuration files for training and evaluation
 ┃ ┣ 📜train_SNN.yml       Main training configuration
 ┃ ┣ 📜eval_MVSEC.yml      Evaluation configuration
 ┃ ┗ 📜parser.py           Configuration parser
 ┃
 ┣ 📂dataloader/           Dataset loading and event encoding
 ┃ ┣ 📜h5.py               HDF5 dataloader for MVSEC dataset
 ┃ ┣ 📜encodings.py        Event encoding methods (voxel grid, event count)
 ┃ ┗ 📜utils.py            Data augmentation utilities
 ┃
 ┣ 📂models/               Neural network architectures
 ┃ ┣ 📜model.py            Main model definitions (FireNet, LIFFireNet, etc.)
 ┃ ┣ 📜spiking_submodules.py    Custom LIF layers
 ┃ ┣ 📜SNNtorch_spiking_submodules.py  SNNtorch-based implementations
 ┃ ┣ 📜submodules.py       Standard ANN building blocks
 ┃ ┗ 📜unet.py             U-Net based architectures
 ┃
 ┣ 📂loss/                 Loss functions and error metrics
 ┃ ┗ 📜flow.py             Event warping loss, AEE, AAE calculations
 ┃
 ┣ 📂utils/                Utility functions
 ┃ ┣ 📜utils.py            Model saving/loading, helper functions
 ┃ ┣ 📜visualization.py    Visualization tools for flow and events
 ┃ ┣ 📜iwe.py              Image of Warped Events (IWE) computation
 ┃ ┣ 📜mlflow.py           MLflow logging utilities
 ┃ ┗ 📜gradients.py        Gradient analysis tools
 ┃
 ┣ 📂tools/                Analysis and debugging scripts
 ┃
 ┣ 📂ONNX_LIF_operator/    Custom ONNX operator for LIF layers
 ┃ ┣ 📜setup.py            Installation script for custom operator
 ┃ ┗ 📂src/                C++ implementation of LIF kernel
 ┃
 ┣ 📜train_flow.py         ⭐ Main training script
 ┣ 📜eval_flow.py          ⭐ Main evaluation script
 ┃
 ┣ 📜train_flow_quant.py   Training with quantization-aware training (QAT)
 ┣ 📜train_flow_validation.py  Training with validation split
 ┣ 📜eval_flow_quant.py    Evaluation for quantized models
 ┃
 ┣ 📜Model_export.py       Model export to ONNX format
 ┣ 📜Model_export_RealQuant.py  INT8 quantized ONNX export (DeepQuant)
 ┣ 📜LIF_layer_export.py   LIF layer export utilities
 ┣ 📜ConvLIF_layer_export.py  Convolutional LIF export
 ┃
 ┗ 📜requirements.txt      Python dependencies
```

### Module Overview

- **configs/**: YAML configuration files defining model architecture, training hyperparameters, and dataset paths
- **dataloader/**: Handles loading event data from HDF5 files and encoding events into voxel grids or event count tensors
- **models/**: Spiking neural network implementations (LIFFireNet, LIFFireFlowNet, etc.)
- **loss/**: Implements photometric consistency loss via event warping and various error metrics (AEE, AAE, etc.)
- **utils/**: Helper functions for model management, visualization, and MLflow experiment tracking
- **ONNX_LIF_operator/**: Custom C++ kernel to enable ONNX export of LIF layers for deployment 
---

## 🔧 Environment Setup

The project requires Python 3.11 and uses conda for environment management. All dependencies are listed in [requirements.txt](requirements.txt).

### Installation Steps

```bash
# Create and activate a new conda environment
conda create -n event_flow python=3.11
conda activate event_flow

# Install pip within the environment
conda install pip

# Install all required packages
pip install -r requirements.txt
```

### Key Dependencies

- **PyTorch 2.7** + torchvision: Deep learning framework
- **snntorch 0.9.4**: Spiking neural network library
- **brevitas**: Quantization library for QAT
- **mlflow 3.2**: Experiment tracking
- **h5py 3.14**: HDF5 file handling for datasets
- **onnx 1.18 + onnxruntime 1.22**: Model export and inference

**Note**: Ensure you run all commands from the repository root directory with the conda environment activated.

---

## 🚀 Quick Start

### 1. Download Dataset

The project uses the **UZH FPV drone dataset** (128×128 resolution) for training and the **MVSEC dataset** (256×256 resolution) for evaluation.

**📥 Download Datasets**:

Both datasets are available from the original source work:
- **[Download UZH FPV & MVSEC Datasets (OneDrive)](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaDBreDBDUktyQVpqeC1FRUl6Zm84aXFCRHJvP2U9VElveEc5&id=19B02A9140C7241D%211951&cid=19B02A9140C7241D)**

After downloading, update the `data.path` field in [configs/train_SNN.yml](configs/train_SNN.yml) and [configs/eval_MVSEC.yml](configs/eval_MVSEC.yml).

```yaml
data:
    path: /path/to/your/dataset/  # Update this
```

**Dataset Resolutions**:
- **MVSEC**: 256×256 (standard for evaluation)
- **UZH FPV**: 128×128 (standard for training)

### 2. Train a Model

**Basic training** (recommended for first-time users):

```bash
python train_flow.py --config configs/train_SNN.yml
```

This will:
- Train a `LIFFireNet` model (default in config)
- Save checkpoints to `mlruns/`
- Log metrics via MLflow
- Display training progress every epoch

**Optional arguments**:
```bash
# Continue training from a previous run
python train_flow.py --config configs/train_SNN.yml --prev_runid <mlflow_run_id>

# Specify MLflow tracking directory
python train_flow.py --config configs/train_SNN.yml --path_mlflow ./custom_mlruns
```

### 3. Evaluate a Trained Model

```bash
python eval_flow.py --config configs/eval_MVSEC.yml --runid <mlflow_run_id>
```

This will:
- Load the model from the specified run
- Evaluate on the test set
- Compute error metrics (AEE, AAE, etc.)
- Optionally visualize results if `vis.enabled: True` in config
- Apply average pooling if `loader.resolution` < `loader.std_resolution` to match target resolution

**Enable visualization**:
```bash
python eval_flow.py --config configs/eval_MVSEC_visual.yml --runid <mlflow_run_id>
```

**Note**: The MVSEC dataset has a native resolution of 256×256. If you evaluate at 128×128, average pooling will automatically downsample the inputs.

### 4. Export Model to ONNX

```bash
python Model_export.py --runid <mlflow_run_id>
```

Exported models are saved to `exported_models/`.

---

## 📜 Main Scripts

These are the primary scripts for the standard workflow:

### Training

| Script | Purpose | Use Case |
|--------|---------|----------|
| **`train_flow.py`** ⭐ | Standard training | **Primary training script** - Start here |
| `train_flow_quant.py` | Quantization-Aware Training (QAT) | Training models for hardware deployment with quantization |
| `train_flow_validation.py` | Training with validation split | Training with separate validation set monitoring |

### Evaluation

| Script | Purpose | Use Case |
|--------|---------|----------|
| **`eval_flow.py`** ⭐ | Standard evaluation | **Primary evaluation script** - Compute metrics on test set |
| `eval_flow_quant.py` | Quantized model evaluation | Evaluate quantized models with calibration support |

### Export

| Script | Purpose | Use Case |
|--------|---------|----------|
| `Model_export.py` | Full model ONNX export | Export complete model for deployment |
| `Model_export_RealQuant.py` | INT8 quantized ONNX export | Export with real INT8 quantization using DeepQuant |
| `LIF_layer_export.py` | LIF layer export utility | Export individual LIF layers |
| `ConvLIF_layer_export.py` | ConvLIF layer export utility | Export convolutional LIF blocks |

---

## 🧠 Model Architectures

The repository implements multiple spiking neural network (SNN) variants for optical flow estimation:

### SNN Models (Spiking Neural Networks)

| Model | Description | Layers | Recurrence |
|-------|-------------|--------|------------|
| **`LIFFireNet`** ⭐ | Spiking FireNet (recommended) | 8 | ✅ LIF-based |
| `LIFFireNet_short` | Compact spiking FireNet | 6 | ✅ LIF-based |
| `LIFFireFlowNet` | Feed-forward only spiking | 8 | ❌ No recurrence |
| `LIFFireFlowNet_short` | Compact feed-forward spiking | 6 | ❌ No recurrence |

**Note**: All models use **LIF (Leaky Integrate-and-Fire)** neurons for ultra-low-power event-based processing.

### Model Selection

To select a model, modify the `model.name` field in your config file:

```yaml
model:
    name: LIFFireNet  # Change this to any model from the table above
    encoding: cnt     # Event encoding: 'voxel' or 'cnt' (count)
    num_bins: 2       # Number of temporal bins for encoding
    base_num_channels: 32
    kernel_size: 3
```

### Spiking Neuron Configuration

LIF neurons can be configured with learnable or fixed parameters:

```yaml
spiking_neuron:
    leak: [0.0, 1.0]      # [min, max] leak rate
    thresh: [0.0, 0.8]    # [min, max] firing threshold
    learn_leak: True      # Learn leak during training
    learn_thresh: True    # Learn threshold during training
    hard_reset: True      # Hard vs soft reset after spike
```

---

## ⚙️ Configuration

All training and evaluation parameters are controlled via YAML config files in [configs/](configs/).

### Main Configuration Files

- **`train_SNN.yml`**: Training configuration (architecture, hyperparameters, dataset)
- **`eval_MVSEC.yml`**: Evaluation configuration (metrics only)
- **`eval_MVSEC_visual.yml`**: Evaluation with visualization enabled

### Key Configuration Sections

#### 1. Dataset Configuration

```yaml
data:
    path: /path/to/dataset/
    mode: events          # 'events' or 'frames'
    window: 1000          # Event window size (number of events)
    window_loss: 10000    # Window size for loss computation
```

#### 2. Model Configuration

```yaml
model:
    name: LIFFireNet
    encoding: cnt         # Event encoding method
    num_bins: 2
    base_num_channels: 32
    kernel_size: 3
    activations: [arctanspike, arctanspike]  # [feed-forward, recurrent]
    mask_output: True     # Mask invalid regions
    quantization:
        enabled: True     # Enable quantization
        PTQ: False        # Post-Training Quantization (False = QAT)
        Conv_only: False  # Quantize only convolutions
```

#### 3. Training Configuration

```yaml
optimizer:
    name: SGD
    lr: 0.0001
    momentum: 0.9
    weight_decay: 0.0001
    nesterov: True

loader:
    n_epochs: 50
    batch_size: 4
    resolution: [128, 128]  # [height, width] - Target processing resolution
    std_resolution: [128, 128]  # [height, width] - Original dataset resolution
    augment: ["Horizontal", "Vertical", "Polarity"]
    augment_prob: [0.5, 0.5, 0.5]
    gpu: 0
```

**Resolution Parameters**:
- **`resolution`**: Target resolution for model processing
- **`std_resolution`**: Original resolution of the dataset
  - **MVSEC dataset**: [256, 256]
  - **UZH FPV dataset**: [128, 128]
- **During evaluation**: If `resolution` < `std_resolution`, average pooling is automatically applied to downsample to the target resolution

```yaml

loss:
    flow_regul_weight: 0.001  # Flow smoothness regularization
    clip_grad: 1.0            # Gradient clipping (null to disable)
```

#### 4. Visualization

```yaml
vis:
    enabled: False       # Enable during evaluation for visual output
    verbose: True        # Print detailed progress
    px: 400             # Visualization resolution
    store_grads: False  # Save gradient statistics
```

### Hot Pixel Filtering

Remove noisy pixels from event data:

```yaml
hot_filter:
    enabled: False
    max_px: 100         # Max number of hot pixels to filter
    min_obvs: 5         # Minimum observations to consider pixel hot
    max_rate: 0.8       # Max event rate threshold
```

------

## 📦 ONNX Export

### Overview

The repository supports exporting models to ONNX format for deployment. The LIF spiking layers require a **custom ONNX operator** implemented in C++.

### Prerequisites

To rebuild or modify the custom LIF operator, you need:

- **Ubuntu 24.04 LTS** (or compatible Linux distribution)
- **libtorch** (matching your PyTorch version)
- **onnxruntime** (compatible with your system)
- **CMake** and build tools

### Setup Instructions

#### 1. Create a Dedicated Environment

It's recommended to use a separate conda environment with matching PyTorch and libtorch versions:

```bash
# Create environment with Python 3.11
conda create -n onnx_export python=3.11
conda activate onnx_export

# Install PyTorch (CPU version for compatibility)
pip install torch==2.9.1+cpu
```

#### 2. Download libtorch

Match the version to your PyTorch installation:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip
unzip libtorch-shared-with-deps-2.9.1+cpu.zip
rm libtorch-shared-with-deps-2.9.1+cpu.zip
```

#### 3. Install onnxruntime

Download the appropriate version from the [onnxruntime releases page](https://github.com/microsoft/onnxruntime/releases).

#### 4. Build the Custom LIF Operator

The operator is already built, but if you need to rebuild:

```bash
cd ONNX_LIF_operator/build/
rm -rf *
cmake ..
make
```

This compiles the kernel from [ONNX_LIF_operator/src/lif_op.cpp](ONNX_LIF_operator/src/lif_op.cpp).

#### 5. Install into PyTorch

**Important**: Before running [setup.py](ONNX_LIF_operator/setup.py), update the library paths to match your system:

```python
# In ONNX_LIF_operator/setup.py
library_dirs = [
    '/path/to/your/libtorch/lib',        # Update this
    '/path/to/your/onnxruntime/lib',     # Update this
]
```

Then install:

```bash
conda activate onnx_export
cd ONNX_LIF_operator/
python setup.py install
```

### Exporting Models

Once the custom operator is installed:

```bash
# Export standard model
python Model_export.py --runid <mlflow_run_id>

# Export individual LIF layer (for testing)
python LIF_layer_export.py

# Export ConvLIF layer (for testing)
python ConvLIF_layer_export.py
```

**Note**: The export scripts automatically use the custom ONNX operator instead of the regular SNNtorch modules during export.

### INT8 Quantized Export (DeepQuant)

For hardware deployment with true INT8 quantization, use `Model_export_RealQuant.py`:

```bash
python Model_export_RealQuant.py <mlflow_run_id> --config configs/eval_MVSEC.yml
```

This script uses **DeepQuant** (required dependency) to export ONNX models with real INT8 precision instead of floating-point quantization-aware operations. Key features:

- **True INT8 tensors**: Quantizes weights and activations to 8-bit integers
- **Calibration-based quantization**: Uses Post-Training Quantization (PTQ) with calibration data
- **Brevitas integration**: Leverages Brevitas quantization annotations for optimal conversion
- **ONNX QDQ format**: Exports models with QuantizeLinear/DequantizeLinear operators

**Requirements**:
- DeepQuant library (for `exportBrevitas` function)
- Brevitas (already in requirements.txt)
- QAT-trained model or model with quantization configuration

The exported model (`exported_models/4_model_dequant_moved.onnx`) is optimized for deployment on edge devices and hardware accelerators supporting INT8 inference.

### Exported Model Location

Exported models are saved to:
```
exported_models/
├── model_<runid>.onnx
└── ...
```

### Testing Exported Models

You can verify the exported ONNX model using:

```bash
# Using netron for visualization
netron exported_models/model_<runid>.onnx

# Using the test script
cd ONNX_LIF_operator/test/
python test_lif_op.py
```

---

## 📊 Experiment Tracking

The project uses **MLflow** for experiment tracking:

- **Tracking directory**: `mlruns/`
- **Metrics logged**: Training loss, AEE, AAE, learning rate
- **Artifacts**: Model checkpoints, configuration files

### View Experiments

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Accessing Run Information

Each training run gets a unique run ID. Use it to:
- Load trained models for evaluation
- Continue training from a checkpoint
- Export models

---

## 🎯 Error Metrics

The evaluation scripts compute several optical flow error metrics:

- **AEE** (Average Endpoint Error): Mean Euclidean distance between predicted and ground truth flow
- **AAE** (Average Angular Error): Mean angular error between flow vectors
- **NEE** (Normalized Endpoint Error): Endpoint error normalized by flow magnitude
- **NAAE** (Normalized Average Angular Error): Weighted angular error
- **AE_ofMeans**: Endpoint error computed on mean flows
- **AAE_Weighted** & **AAE_Filtered**: Variants with weighting/filtering

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory during training**
- Reduce `loader.batch_size` in config (we used 8 for training, and 1 for evaluation)
- Reduce `loader.resolution`

**2. Dataset not found**
- Verify `data.path` in config points to correct directory
- Ensure HDF5 files are in the expected format

**3. MLflow run ID not found**
- Check `mlruns/` directory
- View available runs: `mlflow ui`

**4. ONNX export fails**
- Ensure custom LIF operator is properly installed
- Verify PyTorch and libtorch versions match
- Check library paths in `ONNX_LIF_operator/setup.py`

**5. Import errors for brevitas**
- Some brevitas deprecation warnings are expected
- Ensure `brevitas` is installed: `pip install brevitas`

**6. Understanding resolution parameters**
- `loader.resolution`: Target processing resolution for your model
- `loader.std_resolution`: Original dataset resolution (MVSEC: 256×256, UZH FPV: 128×128)
- During evaluation, if `resolution` < `std_resolution`, average pooling automatically downsamples inputs
- Match `std_resolution` to your dataset for optimal results

---

## 📁 Project Organization Notes

### Redundancy & Cleanup

The repository contains multiple script variants for different use cases:

- **Standard workflow**: `train_flow.py` + `eval_flow.py`
- **Quantization workflow**: `train_flow_quant.py` + `eval_flow_quant.py`
- **Experimental**: `train_flow_validation.py` (validation split variant)

All scripts are kept for backward compatibility and different research needs. **For most users, the standard workflow is recommended.**

### Generated Directories

During usage, the following directories are created:
- `mlruns/`: MLflow experiment tracking data
- `runs/`: TensorBoard logs (for QAT training)
- `exported_models/`: ONNX exported models
- `results_inference/`: Evaluation outputs
- `plots/`: Generated visualizations
- `build/`: CMake build files

---

## 📝 Citation

### Original Work

This implementation is based on the work by Hagenaars, Paredes-Vallés, and de Croon:

```bibtex
@article{hagenaarsparedesvalles2021ssl,
  title={Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks},
  author={Hagenaars, Jesse and Paredes-Vall\'es, Federico and de Croon, Guido},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

### Project Methodology

This project maintains consistency with the original work in terms of:
- Training specifications and hyperparameters
- Dataset usage (MVSEC)
- Network architectures (FireNet-based)

Extensions include quantization support (QAT/PTQ), ONNX export capabilities, and additional model variants.

---

## 📚 Additional Resources

- **Original Paper**: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021)
- **Datasets Download**: [UZH FPV & MVSEC (OneDrive)](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaDBreDBDUktyQVpqeC1FRUl6Zm84aXFCRHJvP2U9VElveEc5&id=19B02A9140C7241D%211951&cid=19B02A9140C7241D)
- **MVSEC Dataset Info**: [Event Camera Dataset](https://daniilidis-group.github.io/mvsec/)
- **SNNtorch Documentation**: [snntorch.readthedocs.io](https://snntorch.readthedocs.io/)
- **Brevitas Documentation**: [github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)

---

**Last Updated**: February 2026
