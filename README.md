## README

This repository contains the core implementation of our **3D multi-sequence fusion network for lumbar spine MRI classification of osteoporosis**, developed for our research on multi-modal MRI analysis.
 The project provides clean and modular code for **data preprocessing, 3D EfficientNet-based modeling and multi-sequence fusion**.

All components are implemented in **PyTorch** and designed for reproducibility and extension.

## 1.Directory Structure

```bash
Directory/
│── dataset.py        # Data loading, preprocessing, augmentation
│── convertNii.py     # DICOM to NIfTI conversion and sequence extraction tool
│── model.py      	  # 3D EfficientNet-B0 backbone + cross-modal fusion module + classifier
│── requirements.txt  # Required Python packages
|── Usage			  # Usage
```



## 2.dataset.py — Data Processing & Loading

This module implements:

- Loading paired **T1/T2 NIfTI volumes**
- Resampling to **128×128×128**
- Intensity clipping & z-score normalization
- Optional N4 bias-field correction
- 3D online augmentation
  (rotation, flip, scaling, elastic deformation)



## 3.convertNii.py — DICOM Sequence Extraction & NIfTI Conversion

This module extracts sequence information from raw DICOM files, categorizes and selects the optimal T1/T2 series, ensures completeness and consistency, and finally converts the validated series into NIfTI format for downstream processing.



## 4.model.py — Network Architecture

This file contains the full implementation of our proposed model:

- A **3D EfficientNet-B0** backbone for each modality
- **Cross-modal attention fusion** between T1 and T2
- Feature combination (multiplication, addition, concatenation)
- Global average pooling (GAP)
- A fully-connected classification head producing **logits**

The model **outputs logits**, not Softmax probabilities, to match PyTorch’s standard training workflow (CrossEntropyLoss).


## 

## 5.requirements.txt

- Required Python packages



## 6.Usage

**1. Prepare the dataset**

- Use `convertNii.py` to process raw DICOM files:
   extract sequence information → classify T1/T2 series → ensure completeness → convert to `.nii.gz`.
- Organize subjects into **normal / osteopenia / osteoporosis** folders.
- Place each subject’s processed **T1** and **T2** volumes into the corresponding modality subfolders.
- Ensure each subject contains a **valid T1–T2 pair**.

**2. Check data paths in `dataset.py`**

- Update `root_path`, class names, and folder structure based on your dataset organization.
- Adjust preprocessing options such as resampling size, intensity normalization, or augmentation settings.

**3. Integrate into your own training pipeline**

- This repository does **not** include training scripts.
   Import `dataset.py` and `model.py` into your custom training code as needed.

```bash
Data Format (After Preprocessing)
After preprocessing, the dataset directory should be organized by class labels and further grouped into T1 and T2 modalities. Each subject must appear in both T1 and T2 folders with matching filenames to ensure paired multimodal training.

dataset_root/
│
├── normal/
│     ├── T1/
│     │     ├── zhangsan/
│     │     │     └── T1_11slices_202506_030813.nii.gz
│     │     ├── wangwu/
│     │     │     └── T1_13slices_202601_172530.nii.gz
│     │     └── ...
│     │
│     ├── T2/
│           ├── zhangsan/
│           │     └── T2_11slices_202506_030814.nii.gz
│           ├── wangwu/
│           │     └── T2_13slices_202601_172531.nii.gz
│           └── ...
│     
│
├── osteopenia/
│     ├── T1/
│     ├── T2/   
│
└── osteoporosis/
      ├── T1/
      ├── T2/
```
