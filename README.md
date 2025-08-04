# EEG-FM-Bench

A comprehensive benchmark platform for EEG (Electroencephalography) foundation models, designed to evaluate and compare the performance of various deep learning models on EEG signal analysis tasks.

## 🌟 Key Features

- **Multi-Model Support**: Integrates multiple state-of-the-art EEG analysis models
  - BENDR
  - BIOT
  - CBraMod
  - Conformer
  - EEGNet
  - EEGPT
  - LaBraM

- **Standardized Data Processing**: Unified data preprocessing and postprocessing pipelines
- **Flexible Configuration System**: YAML-based configuration management
- **Distributed Training**: Support for multi-GPU and multi-node training
- **Visualization Tools**: Model analysis and result visualization capabilities
- **Cluster Support**: Includes SLURM cluster job submission scripts

## 📁 Project Structure

```
EEG-FM-Bench/
├── assets/            # Configuration files and resources
├── baseline/           # Baseline model implementations
│   ├── abstract/      # Abstract base classes
│   ├── bendr/         # BENDR model
│   ├── biot/          # BIOT model
│   ├── cbramod/       # CBRAMod model
│   ├── conformer/     # Conformer model
│   ├── eegnet/        # EEGNet model
│   ├── eegpt/         # EEGPT model
│   └── labram/        # LABRAM model
├── common/            # Common utilities and configurations
├── data/              # Data processing modules
│   ├── dataset/       # Dataset definitions
│   └── processor/     # Data preprocessors
├── plot/              # Visualization tools
└── slurm/             # Cluster job scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. **Project Path**
Change project path to your setting in `./common/conf.py`

### 3. Data Preprocessing

```bash
python preproc.py conf_file=path/to/preproc_config.yaml
```

### 4. Model Training

```bash
python baseline_main.py conf_file=path/to/model_config.yaml model_type=eegpt
```

### 5. Result Visualization

```bash
python visualize.py vis_type=t_sne model_config=path/to/model_config.yaml vis_config=path/to/vis_config.yaml 
```

## ⚙️ Configuration

The project uses YAML configuration files to manage experimental parameters. Example configuration files are located in the `assets/conf/example/` directory.

### Main Configuration Options:

- **Data Configuration**: Dataset paths, batch size, number of workers
- **Model Configuration**: Model parameters, pretrained weight paths
- **Training Configuration**: Learning rate, optimizer, scheduler parameters
- **Logging Configuration**: Output directories, checkpoint saving, cloud synchronization


## 🎯 Visualization Features

- **t-SNE Visualization**: Feature space distribution
- **Integrated Gradients**: Feature importance analysis

## 🖥️ Cluster Usage

The project provides SLURM job scripts for running on high-performance computing clusters:

```bash
# Preprocessing job
sbatch slurm/preproc_submit.slurm conf_file=config.yaml

# Training job
sbatch slurm/baseline_submit.slurm conf_file=config.yaml model_type=eegnet
```

## 📝 Important Notes

1. **Path Configuration**: Please update all path placeholders `/path/to/your/code` in configuration files before first use
2. **GPU Memory**: Some large models may require high-capacity GPU memory
3. **Data Download**: Some datasets require separate application and download

## 🤝 Contributing

We welcome issue reports and feature requests. To contribute code, please:

1. Fork this repository
2. Create a feature branch
3. Submit code changes
4. Create a Pull Request
