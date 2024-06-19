## Requirements
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
