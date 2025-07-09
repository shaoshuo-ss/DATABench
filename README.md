# DATABench

This is the official code repository for the paper titled "[DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective](https://arxiv.org/abs/2507.05622)". DATABench is a comprehensive benchmark for evaluating dataset auditing methods in deep learning from an adversarial perspective. It provides a **unified framework** with **standardized interfaces** for both dataset auditing algorithms and potential attacks, facilitating fair comparison and extensible research in this field.

## Features
- **Unified Evaluation Framework**: Standardized evaluation pipeline for dataset auditing methods under adversarial settings.
- **Extensible Architecture**: Well-defined interfaces for implementing new auditing algorithms and attack methods.
- **Three-Stage Attack Pipeline**: Comprehensive attack framework covering preprocessing, training, and postprocessing stages.
- **ImageFolder Support**: Compatible with any ImageFolder-structured dataset.
- **Rich Attack Arsenal**: Multiple built-in attacks including filtering, differential privacy, adversarial training, and hybrid approaches.
- **Multiple Auditing Methods**: Support for various auditing techniques including MIA, DVBW, DW, and more.



## Project Structure

```
DATABench/
├── audit/                  # Dataset auditing implementations
│   ├── dataset_audit.py   # Base auditing interface
│   ├── MIA.py            # Membership Inference Attack
│   ├── DVBW.py           # Dataset Ownership Verification via Backdoor Watermarking
│   └── ...               # Other auditing methods
├── attack/                # Attack implementations
│   ├── attack_interface.py # Base attack interfaces
│   ├── attack.py         # Attack factory and configuration
│   ├── preprocessing/    # Preprocessing attacks
│   ├── training/         # Training-time attacks
│   └── postprocessing/   # Postprocessing attacks
├── config/               # Configuration files
├── scripts/              # Evaluation scripts
├── utils/                # Utility functions
└── audit_main.py         # Main evaluation script
```



## Quick Start

- **Guidebook about dataset auditing methods** in DATABench: please refer to [audit](audit/README.md).
- **Guidebook about attacks** in DATABench: please refer to [attack](attack/README.md).


### Environment Setup

First, prepare the environment:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Take CIFAR-10 as an example. Transform the dataset into ImageFolder format:

```bash
python utils/transform_cifar10.py
```

**Note**: DATABench supports any ImageFolder-structured dataset.

### Running Evaluations

Execute evaluations using the provided scripts. Config files are located in config:

```bash
bash scripts/audit/DVBW/resnet18-cifar10.sh ${gpus} ${attack} # for evasion attack
bash scripts/forgery/forgery.sh ${gpus} ${audit_method} # for forgery attack
```


## Citation

If you use DATABench in your research, please cite our paper:

```bibtex
@article{shao2025databench,
  title={DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective},
  author={Shao, Shuo and Li, Yiming and Zheng, Mengren and Hu, Zhiyang and Chen, Yukun and Li, Boheng and He, Yu and Guo, Junfeng and Zhang, Tianwei and Tao, Dacheng and Qin, Zhan},
  journal={arxiv preprint arxiv:2507.05622},
  year={2025}
}
```
