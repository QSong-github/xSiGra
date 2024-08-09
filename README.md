# xSiGra: Explainable Single-cell spatial elucidation through image-augmented graph transformer

## Overview
We propose xSiGra, an interpretable graph-based AI model, designed to elucidate interpretable features of identified spatial cell types, by harnessing multi-modal features from spatial imaging technologies. xSiGra employs hybrid graph transformer model to spatial cell type identification by constructing a spatial cellular graph with immunohistology images and gene expression as node attributes. xSiGra uses a novel variant of Grad-CAM component to uncover interpretable features, including pivotal genes and cells for various cell types facilitating deeper biological insights from spatial data.

xSiGra is built using pytorch
Test on: Red Hat Enterprise Linux Server 7.9 (Maipo), NVIDIA Tesla V100 GPU, Intel(R) Xeon(R) CPU E5-2680 v3, 2.50GHZ, 12 core, 64 GB, python 3.10.9, R 4.2.1, CUDA environment(cuda 11.7)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [Tutorial for xSiGra](#tutorial-for-xsigra)
- [Pre-processing scripts](#pre-processing-scripts)
- [Reproduction instructions](#reproduction-instructions)

## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```
Check the list in following section:

- [Requirements](Requirements.md)

## Installation

Download xSiGra:
```
git clone https://github.com/asbudhkar/xSiGra
```

## Dataset
### NanoString CosMx SMI 
The dataset can be download [here](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
### 10x Visium 
The dataset can be download [here](https://github.com/LieberInstitute/HumanPilot/)
### Human Breast Cancer 
The dataset can be download [here](https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0)
### Mouse Brain Anterior
The dataset can be download [here](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-2-sagittal-anterior-1-standard-1-0-0)
### Mouse Brain Coronal
The dataset can be download [here](https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard)

## Folder structure

Check the required folder structure in following section:

- [Folder structure](Requirements.md)

## Tutorial for xSiGra
1. Data processing: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial1_xSiGra_preprocess.ipynb)
2. Run xSiGra: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial2_xSiGra_train.ipynb)
3. xSiGra cluster visualization: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial3_xSiGra_visualize.ipynb)
4. Compute explanations: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial4_xSiGra_explain.ipynb)
5. Compute explanations detailed: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial5_xSiGra_explain_detailed.ipynb)
6. Visualize explanations: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/Tutorial6_xSiGra_visualize_explanations.ipynb)

## Pre-processing scripts
```
Go to /path/to/xSiGra/xSiGra_model
# for NanoString CosMx dataset
python3 processing.py --dataset nanostring

# for 10x Visium dataset
python3 processing.py --dataset 10x
```

## Reproduction instructions

## Test using saved checkpoints

Go to /path/to/xSiGra/xSiGra_model

Download the datasets and [checkpoints](https://drive.google.com/drive/folders/1L7N639ad4pvBHAUAs5zppdNoG4jZqXu4?usp=sharing) and put in folders as above.

#### 1. for NanoString CosMx dataset
The results will be stored in "/path/xSiGra/cluster_results_gradcam_gt1"
```
python3 train_nanostring.py --test_only 1 --dataset lung13 --root ../dataset/nanostring/lung13 --save_path ../checkpoint/nanostring_train_lung13 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0
```
And you can use the bash script to test all slices:
```
sh test_nanostring.sh
```
#### 2. for 10x Visium dataset
The results will be stored in "/path/xSiGra/10x_results/"
```
python3 train.py --test_only 1 --lr $lr --epochs $epoch --id 151676 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust --root ../dataset/10x/
```
And you can use the bash script to test all slices:
```
sh test_visium.sh
```

#### 2. for 10x Visium dataset
The results will be stored in "/path/xSiGra/10x_results/"
```
python3 train.py --test_only 1 --lr $lr --epochs $epoch --id 151676 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust --root ../dataset/10x/
```

#### 3. for Human breast cancer
```
python3 train.py --dataset human_breast_cancer --test_only 1
```

#### 4. for Mouse brain anterior
```
python3 train.py --dataset mouse_brain_anterior --test_only 1
```

#### 5. for Mouse brain coronal
```
python3 train.py --dataset mouse_brain_coronal --test_only 1
```

## Visualize results using saved model

Go to /path/to/xSiGra/xSiGra_model

Download the datasets and [checkpoints](https://drive.google.com/drive/folders/1L7N639ad4pvBHAUAs5zppdNoG4jZqXu4?usp=sharing) and put in folders as above.
Download the [lung13_adata_pred.h5ad](https://drive.google.com/file/d/14FfrqmoFy4md84xentqd1wwsO7mQdj6C/view?usp=sharing) and put in saved_adata folder as above.

#### 1. for NanoString CosMx dataset
The results will be stored in "/path/xSiGra/cluster_results_gradcam_gt1"
```
python3 visualize_nanostring.py --test_only 1 --dataset lung13 --root ../dataset/nanostring/lung13 --save_path ../checkpoint/nanostring_train_lung13 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0
```
And you can use the bash script to test all slices:
```
sh visualize_nanostring.sh
```

## Train from scratch

### Training tutorials

#### 1. for NanoString CosMx dataset
```
python3 train_nanostring.py --dataset lung13 --root ../dataset/nanostring/lung13 --save_path ../checkpoint/nanostring_train_lung13 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0
```

#### 2. for 10x Visium dataset
```
python3 train.py --lr $lr --epochs $epoch --id 151676 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust --root ../dataset/10x/
```

And you can use the bash script to train all slices:
```
sh train_visium.sh
```

#### 3. for Human breast cancer
```
python3 train.py --dataset human_breast_cancer
```

#### 4. for Mouse brain anterior
```
python3 train.py --dataset mouse_brain_anterior
```

#### 5. for Mouse brain coronal
```
python3 train.py --dataset mouse_brain_coronal
```

## Benchmark Explanations

Go to /path/to/xSiGra/explanability_benchmarking

Download the [lung13_adata_pred.h5ad](https://drive.google.com/file/d/14FfrqmoFy4md84xentqd1wwsO7mQdj6C/view?usp=sharing) and put in saved_adata as above or use the anndata computed from testing step above.

#### 1. for NanoString CosMx dataset
The results will be stored in "/path/to/xSiGra/cluster_results_{benchmark_name}_gt1"
```
python3 compute_explanations.py --test_only 1 --dataset lung13 --root ../dataset/nanostring/lung13 --save_path ../checkpoint/nanostring_train_lung13 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0

python3 explainability_benchmarks.py --benchmark deconvolution
```
And you can use the bash script to compute explanations for different benchmarks for lung13. Change the input parameters to compute explanations for other lung cancer slides
```
sh compute_explanations.sh
```

## Evaluate Explanations

Go to /path/to/xSiGra/explanability_evaluation

Two metrics are used: Fidelity and Contrastivity

Download the [lung13_adata_pred.h5ad](https://drive.google.com/file/d/14FfrqmoFy4md84xentqd1wwsO7mQdj6C/view?usp=sharing) and put in saved_adata as above or use the anndata computed from testing step above.
The explanations for different benchmarks need to be computed first and stored in folder structure as explained in above step 

#### 1. for NanoString CosMx dataset
The results for lung13  will be stored in "/path/to/xSiGra/explainability_evaluation/fidelity_results",  "/path/to/xSiGra/explainability_evaluation/fidelity_plots",  "/path/to/xSiGra/explainability_evaluation/contrastivity_plots"


```
# Fidelity

python3 compute_fidelity_all.py --benchmark deconvolution
python3 compute_fidelity_mask.py --benchmark deconvolution

# Contrastivity

python3 evaluate_contrastivity.py

```
And you can use the bash script to compute fidelity for all benchmarks for lung13. Change the input parameters to compute explanations for other lung cancer slides
```
sh compute_fidelity.sh
```

## Downstream Analysis

Go to /path/to/xSiGra/downstream_analysis

Gene enrichment analysis and cell-cell interaction analysis is performed

Download the [lung13_adata_pred.h5ad](https://drive.google.com/file/d/14FfrqmoFy4md84xentqd1wwsO7mQdj6C/view?usp=sharing) and put in saved_adata as above or use the anndata computed from testing step above.
The explanations for different benchmarks need to be computed first and stored in folder structure as explained in above step 
Download [xSiGra explanations](https://drive.google.com/drive/folders/1Ii2v3OzF48ZH5fSehNDSgCh1GMYm61OT?usp=share_link) 

#### 1. for NanoString CosMx dataset
The results for lung13 will be stored in "/path/to/xSiGra/downstream_analysis/enrichment_results". Change the input parameters to compute explanations for other lung cancer slides
The results can be further analysed using softwares like GSEA (Gene Set Enrichment Analysis Workbench) or scanpy for further analysis to gain biological insights.

```
# Gene enrichment analysis

python3 gene_enrichment.py

# Cell-Cell interaction analysis

python3 lr_significant.py
python3 lr_significant1.py
python3 lr_significant2.py
python3 lr_significant3.py

python3 test_significant_pairs.py
```

## Cite

Please cite our paper if you use this code in your own work
```
Aishwarya Budhkar, Ziyang Tang, Xiang Liu, Xuhong Zhang, Jing Su, Qianqian Song, xSiGra: explainable model for single-cell spatial data elucidation, Briefings in Bioinformatics, Volume 25, Issue 5, September 2024, bbae388, https://doi.org/10.1093/bib/bbae388

```
