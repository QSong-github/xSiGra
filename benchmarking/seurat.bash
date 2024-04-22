#!/bin/bash
Rscript run_seurat.R ../dataset/nanostring/lung5-rep1/ 30 8 ./cluster_results_seurat/ lung5-1
Rscript run_seurat.R ../dataset/nanostring/lung5-rep2/ 30 8 ./cluster_results_seurat/ lung5-2
Rscript run_seurat.R ../dataset/nanostring/lung5-rep3/ 30 8 ./cluster_results_seurat/ lung5-3
Rscript run_seurat.R ../dataset/nanostring/lung6/ 30 4 ./cluster_results_seurat/ lung6
Rscript run_seurat.R ../dataset/nanostring/lung9-rep1/ 20 8 ./cluster_results_seurat/ lung9-1
Rscript run_seurat.R ../dataset/nanostring/lung9-rep2/ 45 4 ./cluster_results_seurat/ lung9-2
Rscript run_seurat.R ../dataset/nanostring/lung12/ 28 8 ./cluster_results_seurat/ lung12
Rscript run_seurat.R ../dataset/nanostring/lung13/ 20 8 ./cluster_results_seurat/ lung13