# [ICASSP2025] QuantumMUSE_EEG
[![arXiv](https://img.shields.io/badge/arXiv-2408.13919-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.13919)  [ICASSP2025](https://ieeexplore.ieee.org/document/10889504)

## Overview
This repository contains the implementation of a "Quantum Multimodal Contrastive Learning Framework" for EEG-image data analysis and processing. The work has been successfully accepted for publication at ICASSP 2025.

The framework leverages quantum computing techniques combined with multimodal contrastive learning to enhance EEG signal processing and analysis. By incorporating quantum layers within neural network architectures, the model achieves improved feature extraction and representation learning from EEG data.

![image](https://github.com/user-attachments/assets/d88b39f4-343b-4ff6-ae46-612e04bf74a6)


## Project Structure

### Detailed Directory Structure
```
QuantumMUSE_EEG_2/
├── Data/
│   └── Things-EEG2/
│       ├── Preprocessed_data_250Hz/
│       ├── DNN_feature_maps/
│       └── Image_set/
├── preprocessing/
│   ├── get_center_images.py
│   ├── preprocessing.py
│   ├── preprocessing_utils.py
│   └── __pycache__/
├── model/
│   ├── main_train.py
│   ├── muse_eeg_model.py
│   ├── qcl_train.py
│   ├── test0_1_Proj_img_cls.pth
│   ├── test0_1_Proj_eeg_cls.pth
│   ├── test0_1_Enc_custom_eeg_cls.pth
│   └── __pycache__/
├── clipvit_feature_extraction/
│   ├── center_fea_clip.py
│   ├── feature_maps_clip.py
│   └── obtain_feature_maps_clip.py
├── results/
│   ├── log_subject1.txt
│   └── result.csv
└── README.md
```

### Component Description
- **Data/**: Contains the Things-EEG2 dataset with preprocessed EEG data, DNN feature maps, and image stimuli
- **preprocessing/**: Scripts for EEG data preprocessing and utility functions
- **model/**: Implementation of the quantum multimodal contrastive learning model
  - *main_train.py*: Main training script
  - *muse_eeg_model.py*: MUSE EEG model implementation
  - *qcl_train.py*: Quantum contrastive learning training script
  - *.pth files*: Pre-trained model weights
- **clipvit_feature_extraction/**: Tools for extracting features using CLIP/ViT architectures
- **results/**: Experimental results and model outputs

## Key Features

- **Quantum Neural Networks**: Integration of quantum computing techniques with traditional neural networks
- **Multimodal Contrastive Learning**: Joint learning from EEG signals and visual/textual data
- **CLIP/ViT Feature Extraction**: Leveraging pre-trained vision transformers for feature extraction
- **State-of-the-art Performance**: Improved accuracy in EEG-based analysis and classification

## Technical Details

The framework implements quantum layers using PennyLane, a cross-platform Python library for quantum machine learning. The quantum layers are integrated within traditional neural network architectures to process EEG data.

Key components include:
- Quantum circuit implementation with parameterized gates
- Quantum-enhanced feature extraction
- Multimodal contrastive learning objectives
- Integration with CLIP/ViT for vision-based feature extraction

## Publication

The framework introduced in this repository has been accepted to the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2025 as "Quantum Multimodal Contrastive Learning Framework".

## Requirements

```
torch
pennylane
numpy
scipy
sklearn
matplotlib
open_clip_torch
```

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/yourusername/QuantumMUSE_EEG_2.git
cd QuantumMUSE_EEG_2
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare your data in the Data/ directory

4. Run the training script
```bash
python model/qcl_train.py
```

## Citation

```
@INPROCEEDINGS{10889504,
  author={Chen, Chi-Sheng and Tsai, Aidan Hung-Wen and Huang, Sheng-Chieh},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Quantum Multimodal Contrastive Learning Framework}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Portable document format},
  doi={10.1109/ICASSP49660.2025.10889504}}
```
