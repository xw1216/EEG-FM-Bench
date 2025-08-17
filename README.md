# EEG-FM-Bench: A Comprehensive Benchmark for EEG Foundation Models

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/your-paper-link)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/xw1216/EEG-FM-Bench)
[![Datasets](https://img.shields.io/badge/Datasets-14_Curated-blue)](#-datasets)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

*A comprehensive benchmark for systematic and standardized evaluation of EEG foundation models*

[📁 Project Architecture](#-project-architecture) • [🚀 Quick Start](#-quick-start) • [📊 Datasets](#-datasets) • [🏗️ Models](#️-supported-models) • [📈 Results](#-benchmark-results) • [📖 Documentation](#-documentation) • [🤝 Limitations & Contributing](#-limitations--contributing)

</div>

---

## 🌟 What is EEG-FM-Bench?

EEG-FM-Bench addresses a critical gap in neuroscience AI research: the lack of standardized evaluation frameworks for EEG foundation models. As these models rapidly proliferate, inconsistent evaluation methods have made fair comparisons nearly impossible, hindering scientific progress.

**Our contributions:**
- 🎯 **Unified Benchmark Platform**: First comprehensive framework for standardized EEG-FM evaluation
- 📚 **Diverse Task Suite**: 14 datasets across 10 canonical EEG paradigms 
- 🔬 **Multiple Evaluation Strategies**: Frozen backbone, full fine-tuning, and multi-task learning
- 🎨 **Rich Analysis Tools**: Quantitative metrics + qualitative visualizations (t-SNE, Integrated Gradients)
- 🔄 **Reproducible Science**: Open-source codebase with standardized protocols

## ✨ Key Features

### 🤖 **Foundation Model Support**
Comprehensive evaluation of state-of-the-art EEG foundation models:
- **BENDR** - Transformer with contrastive self-supervised learning
- **BIOT** - Biosignal transformer for cross-data learning  
- **CBraMod** - Criss-cross attention for spatio-temporal modeling
- **EEGPT** - Dual self-supervised universal representation learning
- **LaBraM** - Large brain model with vector quantization
- **EEGConformer** - Hybrid CNN-Transformer architecture for EEG analysis

### 📊 **Comprehensive Dataset Coverage**
| Paradigm | Datasets | Tasks |
|----------|----------|--------|
| **Motor Imagery** | BCIC-2a, PhysioMI, Mimul-11 | 3/4-class imagined movement classification |
| **Emotion Recognition** | SEED, SEED-V, SEED-VII | 3/5/7-class emotion state recognition |
| **Clinical Applications** | TUAB, TUEV, Siena, HMC, TUSL | Abnormal detection, seizure/event classification, sleep staging |
| **Cognitive & Neurodegenerative** | Things-EEG-2, Workload, ADFTD | Visual target detection, mental workload, AD classification |

**Note**: This repository provides the benchmark framework and evaluation code. **Datasets must be downloaded separately** from their original sources due to licensing restrictions.

### 🔧 **Advanced Evaluation Framework**
- **Frozen Backbone**: Evaluate representation quality without task-specific adaptation
- **Full Fine-tuning**: Assess model adaptability to downstream tasks  
- **Multi-task Learning**: Test knowledge sharing across diverse EEG paradigms
- **Standardized Preprocessing**: Unified pipeline (filtering, resampling, segmentation)
- **Robust Metrics**: Balanced accuracy, weighted F1, AUROC, AUC-PR, Cohen's Kappa

### 🎨 **Rich Visualization & Analysis**
- **t-SNE Embeddings**: Visualize learned feature representations
- **Integrated Gradients**: Understand model decision-making processes across different architectures
- **Neurophysiological Validation**: Ensure models focus on relevant brain regions

## 📁 Project Architecture

```
EEG-FM-Bench/
├── assets/                # Configuration templates & resources
│   └── conf/example/      #    Example configs for models & datasets
├── baseline/              # Foundation model implementations  
│   ├── abstract/          #    Base class to be inherited for other models
│   ├── bendr/             #    BENDR: Transformer + contrastive learning
│   ├── biot/              #    BIOT: Cross-data biosignal learning
│   ├── cbramod/           #    CBraMod: Criss-cross attention
│   ├── eegpt/             #    EEGPT: Dual self-supervised learning
│   ├── labram/            #    LaBraM: Vector quantized brain model
│   ├── eegnet/            #    EEGNet: Compact CNN baseline
│   └── conformer/         #    EEGConformer: Hybrid CNN-Transformer 
├── common/                # Shared utilities & configurations
├── data/                  # Data processing ecosystem
│   ├── dataset/           #    14 benchmark dataset definitions
│   └── processor/         #    Standardized preprocessing pipeline
├── plot/                  # Advanced visualization tools
├── slurm/                 # HPC cluster job scripts
├── baseline_main.py       # Main training entry point for model training and evaluation
├── preproc.py             # Data preprocessing pipeline execution script
├── visualize.py           # Visualization generation (t-SNE, Integrated Gradients)
└── requirements.txt       # Python package dependencies 
```

## 🚀 Quick Start

### 📋 Prerequisites

```bash
# Clone the repository
git clone https://github.com/xw1216/EEG-FM-Bench.git
cd EEG-FM-Bench

# Install dependencies
pip install -r requirements.txt
```

### ⚙️ Configuration Setup

1. **Update project paths** in `./common/path.py`:
```python
RUN_ROOT =
LOG_ROOT =
DATABASE_RAW_ROOT =
DATABASE_PROC_ROOT =
DATABASE_CACHE_ROOT =
```

2. **Configure your experiment** using YAML files like examples in `assets/conf/example/`, values not assigned will be set according to pydantic model class:
```yaml
# Example: assets/conf/example/eegpt/eegpt.yaml
model:
  pretrained_path: "/path/to/eegpt/weights"
data:
  batch_size: 128
  num_workers: 4
training:
  max_lr: 1e-4
  max_epochs: 50
log:
  output_dir: "/path/to/run/log"
```

### 🔄 Pipeline Execution

#### Step 1: Dataset Download & Preprocessing
```bash
# First, download datasets from their original sources (see Dataset Guide)
# Then preprocess with standardized pipeline
# Config file can be identified by absolute path or relative path to CONF_ROOT
python preproc.py conf_file=preproc/preproc_remote.yaml
```

#### Step 2: Model Evaluation
```bash
# Fine-Tuning (examples for different models)
python baseline_main.py conf_file=example/eegpt/eegpt.yaml model_type=eegpt
```

#### Step 3: Analysis & Visualization
```bash
# Generate t-SNE embeddings
python visualize.py t_sne example/eegpt/eegpt.yaml /path/to/t_sne_config_eegpt.yaml

# Create integrated gradients analysis
python visualize.py integrated_gradients example/eegpt/eegpt.yaml /path/to/integrated_gradients_config_eegpt.yaml
```

## 📊 Datasets

Our benchmark encompasses 14 carefully curated datasets spanning 10 canonical EEG paradigms. **All datasets must be downloaded separately from their original sources.**

<details>
<summary><b>🧠 Motor Imagery & Movement</b></summary>

- **BCIC-2a**: 4-class classification (left hand, right hand, feet, tongue)
- **PhysioMI**: 4-class motor imagery (left fist, right fist, both fists, feet)
- **Mimul-11**: 3-class upper extremity tasks (reaching, grasping, twisting)
</details>

<details>
<summary><b>😊 Emotion Recognition</b></summary>

- **SEED**: 3-class emotion recognition (sad, neutral, happy)
- **SEED-V**: 5-class emotion states (disgust, fear, sad, neutral, happy)
- **SEED-VII**: 7-class emotion recognition (disgust, fear, sad, neutral, happy, anger, surprise)
</details>

<details>
<summary><b>🏥 Clinical Applications</b></summary>

- **TUAB**: Binary abnormal EEG detection (abnormal vs normal)
- **TUEV**: 6-class epileptiform event classification (spike-wave, GPED, PLED, eye movement, artifact, background)
- **Siena**: Binary seizure detection (seizure vs healthy)
- **HMC**: 5-class sleep stage classification (wake, REM, N1, N2, N3)
- **TUSL**: 3-class slowing event classification (seizure, slow wave, background)
</details>

<details>
<summary><b>🧩 Cognitive & Neurodegenerative</b></summary>

- **Things-EEG-2**: Binary visual target detection (target vs non-target)
- **Workload**: Binary mental workload assessment (arithmetic calculation vs resting)
- **ADFTD**: 3-class dementia classification (Alzheimer's Disease, Frontotemporal Dementia, healthy)
</details>

### 📥 Dataset Acquisition

**Each dataset must be downloaded separately from its original source.** This repository contains only the dataset loaders and preprocessing configurations - no actual data is distributed.

#### 🔍 Finding Dataset Information

Each of our 14 benchmark datasets has a corresponding Python file in `data/dataset/` that contains:

- **📖 Academic Citations**: Proper references for the original papers
- **🔗 Dataset Sources**: Information about where to find and request access to data  
- **⚙️ Preprocessing Configurations**: All technical parameters pre-configured
- **📁 Expected File Directory**: Required directory organization
- **📝 Usage Notes**: Special requirements and considerations

#### 🚀 General Acquisition Process

**Step 1: Explore Available Datasets**
```bash
# Browse all available dataset implementations
ls data/dataset/
find data/dataset/ -name "*.py"
```

**Step 2: Check Dataset Requirements**
```bash
# View dataset class documentation
python -c "
from data.dataset.workload import WorkloadConfig  # Example
conf = WorkloadConfig(name='finetune')
print(conf.description)
print(conf.citation)
"
```

**Step 3: Locate Original Sources**
- Many datasets require **individual applications** or **institutional access**
- **Check the orginal paper** for detailed descriptions and download method for each dataset
- Some datasets may require **data use agreements**

**Step 4: Download & Organize**
```bash
# Follow the directory structure specified in each dataset file
# Example structure (varies by dataset):
DATABASE_RAW_ROOT/
├── dataset_name/
│   └── scan_dir
```

**Step 5: Configure Paths**
```bash
# Update preprocessing configuration with your data paths
vim assets/conf/example/preproc/preproc_remote.yaml
# Edit the YAML file to add your downloaded datasets to preproc list
```

#### ⚠️ Important Considerations

- **No Direct Downloads**: Dataset files contain **source information**, not download links
- **Individual Licensing**: Each dataset has **unique terms and requirements**
- **Registration Often Required**: Many datasets need **approval before access**
- **Large File Sizes**: Plan for **several GBs per dataset**
- **Directory Structure**: Must **exactly match** the expectations in dataset files
- **Preprocessing Pipeline**: All parameters are **pre-configured** for consistency


## 🏗️ Supported Models

### Foundation Models

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| **BENDR** | Transformer + CNN | Contrastive learning from speech |
| **BIOT** | Channel-independent Transformer | Variable channel tokenization |
| **CBraMod** | Criss-cross Attention | dual-branch spatio-temporal modeling |
| **EEGPT** | Dual-branch Transformer | momentum latent feature alignment |
| **LaBraM** | Vector Quantized VAE + Transformer | Discrete neural codebook |

### Classical Baselines
- **EEGNet**: Compact CNN for EEG classification
- **EEGConformer**: Hybrid CNN-Transformer architecture combining local feature extraction with global attention

## 📈 Benchmark Results

### Key Findings

🔍 **Generalization Gap**: Frozen backbone evaluation reveals that many current foundation models struggle with out-of-the-box transfer, achieving near-random performance on many tasks.

🎯 **Architecture Matters**: Models with EEG-specific designs (CBraMod, EEGPT) consistently outperform generic transformers.

🔄 **Multi-task Magic**: Joint training across paradigms significantly boosts performance, especially for underperforming models.

### Sample Results (Balanced Accuracy %)

| Model | SEED (Emotion) | PhysioMI (MI) | HMC (Sleep) |
|-------|---------------|-----------------|-------------|
| **BENDR** | 59.50±0.42 | 47.78±0.28 | 72.63±0.13 |
| **BIOT** | 63.87±1.77 | 27.38±0.35 | 71.01±0.07 |
| **LaBraM** | 61.59±1.71 | 57.27±0.26 | 69.87±0.10 |
| **EEGPT** | 69.81±0.45 | 54.16±0.18 | 69.67±1.24 |
| **CBraMod** | 70.83±0.28 | 56.74±0.36 | 71.48±0.40 |

*Results shown for separate full fine-tuning strategy. See paper for complete analysis.*

## 🖥️ High-Performance Computing

### SLURM Integration

```bash
# Large-scale preprocessing (after downloading datasets)
sbatch slurm/preproc_submit.slurm conf_file=your_preproc_config.yaml

# Distributed model training (examples for different models)
sbatch slurm/baseline_submit.slurm conf_file=your_model_config.yaml model_type=eegpt
```

### Resource Requirements
- **Preprocessing**: 64GB RAM, 16~32 CPU cores
- **Training**: 1-8 A100 GPUs or better (depending on batch size)
- **Storage**: ~500GB for all datasets (processed, user must download separately)

## 📖 Documentation

### Configuration System

All experiments use YAML configuration files that must match the Pydantic structure defined in `common/config.py` and model-specific config class like `baseline\eegpt\eegpt_config.py`:

<details>
<summary><b>📋 Complete Configuration Example</b></summary>

```yaml
# Configuration file structure (matches Pydantic BaseModel hierarchy)
# Training pattern flags
seed: 42
master_port: 51002
multitask: true
model_type: 'eegpt'

# Data configuration
data:
  batch_size: 32
  num_workers: 1
  datasets:
    tuab: 'finetune'

# EEGPT-specific model configuration
model:
  # Pretrained weights - each model will load from this checkpoint
  pretrained_path: "/path/to/your/ckpt"

  # Channel adaptation
  use_channel_conv: false
  conv_chan_dim: 22

  head_dropout: 0.1
  mlp_hidden_dim: [128]

# Training configuration
training:
  max_epochs: 50

  # Optimizer settings
  lr_schedule: "onecycle"  # 'onecycle' or 'cosine'
  max_lr: 5e-4
  encoder_lr_scale: 0.1    # Scale factor for encoder learning rate
  warmup_epochs: 5
  warmup_scale: 1e-2

  # Training options
  freeze_encoder: false     # Whether to freeze encoder weights
  use_amp: true           # Use automatic mixed precision

  label_smoothing: 0.1    # Label smoothing factor

# Logging configuration
logging:
  experiment_name: "eegpt"
  output_dir: "/path/to/your/log"
  ckpt_dir: "/path/to/save/ckpt"
  
  # Cloud logging configuration
  use_cloud: true
  cloud_backend: "wandb"
  
  # Cloud logging parameters (works for both wandb and comet)
  project: 'eegpt'
  api_key: null      # API key (or set via WANDB_API_KEY environment variables)
  offline: false
  
  tags: ['eegpt']

  # Logging intervals
  log_step_interval: 1      # Log every N steps
  ckpt_interval: 5       # Evaluate and save ckpt every N epochs
```
</details>

### Advanced Usage

<details>
<summary><b>🔧 Custom Dataset Integration</b></summary>

To add a new dataset, create a file in `data/dataset/` and implement the required interface:

```python
@dataclass
class TemplateConfig(EEGConfig):
    name: str = 'finetune'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("0.0.0")
    description: Optional[str] = ""
    citation: Optional[str] = """
    bibtex Citation
    """

    filter_notch: float = 50.0
    is_notched: bool = False

    dataset_name: Optional[str] = 'template'
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
            'Fp1',
            'Fp2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 10
    suffix_path: str = 'template'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['class 1', 'class 2'])


class TemplateBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TemplateConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        pass

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]

```
</details>

<details>
<summary><b>🤖 Custom Model Integration</b></summary>

To add a new foundation model, create a directory in `baseline/` and implement the interfaces in `baseline/abstract`:

```python
# baseline/your_model/your_model_config.py
class YourModelConfig(BaseModelArgs):
    """Model-specific configuration extending BaseModelArgs."""
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    # Add your model-specific parameters with type annotations


# baseline/your_model/model.py
class YourFoundationModel(nn.Module):
    """Your foundation model architecture."""
    
    def __init__(self, encoder, classifier, ):
        super().__init__()
        # Implement your architecture
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        pass


# baseline/your_model/your_model_trainer.py
class YourModelTrainer(AbstractTrainer):
    """Main trainer class for your model."""
    
    def __init__(self, config: YourModelConfig):
        super().__init__(config)
        
    def build_model(self) -> nn.Module:
        """Build the foundation model (include classifier) architecture."""
        return YourFoundationModel(...)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        pass


# baseline/your_model/your_model_adapter.py
# If your model uses different input data format, you should create an DatasetAdapter to do runtime data conversion
class YourModelDatasetAdapter(AbstractDatasetAdapter):
    def _setup_adapter(self):
      """Initialize specific adapter configurations."""
        self.model_name = 'your_model'
        super()._setup_adapter()

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
      """Process a single sample according to model requirements."""
      pass

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by your model"""
        return []

# Integrate DatasetAdapter into DataLaderFactory class
class YourModelDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> YourModelDatasetAdapter:
        return YourModelDatasetAdapter(dataset, dataset_names, dataset_configs) 


# baseline/__init__.py
# Register your own classes to Registry
ModelRegistry.register_model(
    model_type='your_model',
    config_class=YourModelConfig,
    adapter_class=YourModelDataLoaderFactory, # or None if no conversion needed
    trainer_class=YourModelTrainer
)
```

**Required Files:**
- `baseline/your_model/your_model_trainer.py` 
- `baseline/your_model/your_model_config.py`
- `baseline/your_model/model.py`
- `assets/conf/example/your_model/your_model.yaml`

**For reference, see existing model implementations:**
- `baseline/conformer/` - EEGConformer for classic implementation example
- `baseline/eegpt/` - EEGPT for foundation model implementation example
</details>

## 📚 Citations

If you use EEG-FM-Bench in your research, please cite our paper:

```bibtex
@article{eeg-fm-bench2024,
  title={EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models},
  author={[Authors]},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

### Model Citations

When using specific models, please also cite the original papers:

<details>
<summary><b>Foundation Model Citations</b></summary>

```bibtex
@article{kostas2021bendr,
  title={BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data},
  author={Kostas, Demetres and Aroca-Ouellette, Stephane and Rudzicz, Frank},
  journal={Frontiers in Human Neuroscience},
  volume={15},
  pages={653659},
  year={2021},
  publisher={Frontiers Media SA}
}

@article{yang2023biot,
  title={Biot: Biosignal transformer for cross-data learning in the wild},
  author={Yang, Chaoqi and Westover, M and Sun, Jimeng},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={78240--78260},
  year={2023}
}

@article{wang2024cbramod,
  title={Cbramod: A criss-cross brain foundation model for eeg decoding},
  author={Wang, Jiquan and Zhao, Sha and Luo, Zhiling and Zhou, Yangxuan and Jiang, Haiteng and Li, Shijian and Li, Tao and Pan, Gang},
  journal={arXiv preprint arXiv:2412.07236},
  year={2024}
}

@article{wang2024eegpt,
  title={Eegpt: Pretrained transformer for universal and reliable representation of eeg signals},
  author={Wang, Guangyu and Liu, Wenchao and He, Yuhong and Xu, Cong and Ma, Lin and Li, Haifeng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={39249--39280},
  year={2024}
}

@article{jiang2024labram,
  title={Large brain model for learning generic representations with tremendous EEG data in BCI},
  author={Jiang, Wei-Bang and Zhao, Li-Ming and Lu, Bao-Liang},
  journal={arXiv preprint arXiv:2405.18765},
  year={2024}
}
```
</details>

<details>
<summary><b>Classical Baseline Citations</b></summary>

```bibtex
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={iOP Publishing}
}

@article{song2022eeg-conformer,
  title={EEG conformer: Convolutional transformer for EEG decoding and visualization},
  author={Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={31},
  pages={710--719},
  year={2022},
  publisher={IEEE}
}
```
</details>

## 🤝 Limitations & Contributing

We welcome contributions from the community!

### 🚨 Known Limitations

This project was initially developed for personal research purposes and implemented as a single-developer effort. While we've made it available to the community, please be aware of the following limitations:

- **🐛 Bugs & Issues**: As a personal project, you may encounter bugs or inconsistencies in the codebase. The overall design might not always follow best practices or feel convenient for all use cases.

- **🔧 Design Decisions**: Some architectural choices were made to solve specific research problems and may not generalize well to other scenarios. We acknowledge that the framework might need significant refactoring for broader adoption.

- **📦 Missing Model Implementations**: Some foundation models referenced in our paper have not released their official code or pre-trained weights. In these cases, we excluded models entirely when reliable implementation was not feasible.

- **⚡ Reproducibility Challenges**: Due to the above limitations, exact reproduction of all published results may not always be possible. We've done our best to document these cases clearly.

- **🏗️ Single Developer Limitations**: Code style, documentation quality, and API design may be inconsistent across different parts of the codebase.

**We greatly appreciate your understanding and encourage contributions to help improve these limitations!**

### How to Contribute
- 🐛 **Bug Reports**: Open an issue with reproduction steps - these are especially valuable given the current limitations
- 🚀 **Feature Requests**: Propose new models, datasets, or analysis tools  
- 📝 **Documentation**: Improve our guides and examples - documentation PRs are highly welcomed
- 🔬 **Research**: Share your findings and improvements
- 🔧 **Code Quality**: Help refactor and improve the overall codebase design
- 📦 **Model Implementations**: Contribute official implementations of missing foundation models


## 📄 License

This project is licensed under the **Apache License 2.0**.
See the [LICENSE](LICENSE) file for complete terms and conditions.

**Important**: Individual datasets have their own licensing terms. Users must comply with all dataset-specific licenses when downloading and using the data.

## 🙏 Acknowledgments

- **Dataset Providers**: We thank all dataset creators for making their data publicly available. Please cite original dataset papers when using the data.
- **Foundation Model Authors**: Thanks for open-sourcing model implementations that enable fair comparison
- **Research Community**: The neuroscience and BCI communities for inspiration, feedback, and collaboration


---

<div align="center">

**🌟 Star us on GitHub if EEG-FM-Bench helps your research! 🌟**

[⬆ Back to Top](#eeg-fm-bench-a-comprehensive-benchmark-for-eeg-foundation-models)

</div>
