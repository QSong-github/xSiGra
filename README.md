# xSiGra: Explainable Single-cell spatial elucidation through image-augmented graph transformer

xSiGra is built using pytorch
Test on: Red Hat Enterprise Linux Server 7.9 (Maipo), NVIDIA Tesla V100 GPU, Intel(R) Xeon(R) CPU E5-2680 v3, 2.50GHZ, 12 core, 64 GB, python 3.10.9, R 4.2.1, CUDA environment(cuda 11.7)

## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```
```
anndata==0.8.0
arrow==1.2.3
captum==0.6.0
gseapy==1.0.4
holoviews==1.16.2
imageio==2.26.0
imbalanced-learn==0.10.1
imblearn==0.0
jupyter==1.0.0
kaleido==0.1.0
leidenalg==0.9.1
louvain==0.8.0
matplotlib==3.6.3
matplotlib-inline==0.1.6
matplotlib-venn==0.11.9
numpy==1.23.5
pandas==1.5.3
pip==23.1.2
plotly==5.13.1
scanpy==1.9.2
scipy==1.10.1
seaborn==0.12.2
SpaGCN==1.2.5
STAGATE-pyG==1.0.0
statsmodels==0.14.0
stlearn==0.4.11
torch==1.13.1+cu117
torch-geometric==2.2.0
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.16+pt113cu117
torchvision==0.14.1
tqdm==4.64.1
wandb==0.13.10
```
## Installation

Download xSiGra:
```
git clone https://github.com/asbudhkar/xSiGra
```

## Dataset Setting
### NanoString CosMx SMI 
The dataset can be download [here](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
### 10x Visium 
The dataset can be download [here](https://github.com/LieberInstitute/HumanPilot/)

## Folder structure

```
├── requirements.txt
├── dataset
│   └── 10x
│        └── 151676
│              ├── filtered_feature_bc_matrix.h5
│              ├── metadata.tsv 
│              ├── sampledata.h5ad
│              └── spatial
│                     ├── tissue_positions_list.csv  
│                     ├── full_image.tif  
│                     ├── tissue_hires_image.png  
│                     ├── tissue_lowres_image.png
│   └── nanostring
│        └── lung13
│             └── Lung13_exprMat_file.csv
│             └── matched_annotation_all.csv
│             └── fov1
│                   ├── CellComposite_F001.jpg
│                   ├── sampledata.h5ad
│             └── fov2
│                   ├── CellComposite_F002.jpg
│                   ├── sampledata.h5ad
├── checkpoint
│   └── nanostring_train_lung13
│        ├── best.pth
│   └── 10x
│        └── 151507
│              ├── final_0.pth
├── saved_adata
│   └── store computed or our provided checkpoint adata
├── cluster_results_gradcam_gt1
│   └── cluster0_endothelial.csv
│   └── cluster1_epithelial.csv
│   └── cluster2_fibroblasy.csv
│   └── cluster3_lymphocyte.csv
│   └── cluster4_neutrophil.csv
│   └── cluster5_myeloid.csv
│   └── cluster6_neutrophil.csv
│   └── cluster7_tumors.csv
├── cluster_results_deconvolution_gt1
│   └── ...
├── cluster_results_inputxgradient_gt1
│   └── ...
├── cluster_results_guidedbackprop_gt1
│   └── ...
├── cluster_results_saliency_gt1
│   └── ...
├── 10x_results
│   └── ...
├── benchmarking
│   └── cluster_results_bayesspace
│       └── ...
│   └── cluster_results_scanpy
│       └── ...
│   └── cluster_results_seurat
│       └── ...
│   └── cluster_results_spagcn
│       └── ...
│   └── cluster_results_stagate
│       └── ...
│   └── cluster_results_stlearn
│       └── ...
│   └── ...  
├── benchmarking_10x
│   └── 10x_bayesspace
│       └── ...
│   └── 10x_scanpy
│       └── ...
│   └── 10x_seurat
│       └── ...
│   └── 10x_stagate
│       └── ...
│   └── 10x_stlearn
│       └── ...
│   └── ... 
│── explainability_benchmarking
│   └── ...   
├── explainability_evaluation
│   └── contrastivity_plots
│       └── ...
│   └── fidelity_plots
│       └── ...
│   └── fidelity_results
│       └── ...
│   └── ...  
│── downstream_analysis  
│   └── enrichment_results 
│       └── ...
│   └── ...  
│── Tutorials
```

## Tutorial for xSiGra
1. Data processing: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_preprocess.ipynb)
2. Run xSiGra: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_train.ipynb)
3. xSiGra cluster visualization: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_visualize.ipynb)
4. Compute explanations: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_explain.ipynb)
5. Compute explanations detailed: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_explain_detailed.ipynb)
6. Visualize explanations: [here](https://github.com/asbudhkar/xSiGra/blob/main/Tutorials/xSiGra_visualize_explanations.ipynb)

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
```
