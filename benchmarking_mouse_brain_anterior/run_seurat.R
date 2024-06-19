args = commandArgs(trailingOnly=TRUE)
dataroot <- args[1]
file_name <- args[2]
n_clusters <- args[3]
savepath <- args[4]

# Rscript run_seurat.R  ../dataset/breast_invasive_carcinoma/ Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5 8 ./HBC_Seurat

library(SeuratObject,lib="/N/project/st_brain/abudhkar")
library(Seurat,lib="/N/project/st_brain/abudhkar")
library(SeuratData,lib="/N/project/st_brain/abudhkar")
library(SeuratDisk,lib="/N/project/st_brain/abudhkar")
library(ggplot2)
library(patchwork)
library(dplyr)
options(bitmapType = 'cairo')
library(anndata)


show_seurat <- function(input_root, file_name, n_cluster, savepath){
    dir.input = input_root
    output = savepath
    if (!dir.exists(file.path(output))){
        dir.create(file.path(output), recursive = TRUE)
    }

    sp_data <- Load10X_Spatial(dir.input, filename = file_name)
    print(sp_data)
    df_meta <- read.csv(file.path(dir.input, 'spatial/tissue_positions_list.csv'),
                        stringsAsFactors = F, header=F, check.names=F, row.names=1)
    df_meta1 = df_meta[match(colnames(sp_data),rownames(df_meta)),]
    df_meta1 = df_meta1[,-1,drop=FALSE]
    colnames(df_meta1) = c('sx','sy','px','py')
    identical(rownames(df_meta1),colnames(sp_data))
    sp_data@meta.data = cbind(sp_data@meta.data, df_meta1)
    
    # remove nan from sp_data
    sp_data = na.omit(sp_data)
    ab <- colSums(sp_data)
    sp_data@meta.data$sums <- ab
    sp_data <- subset(sp_data,sums!=0)
    
    print('sctransofrom')
    set.seed(1234)
    sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = T, variable.features.n = 2000)
    set.seed(1234)
    sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
    sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:50)
    
    for (resolution in 5:50){
        set.seed(1234)
        sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
        
        if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
            break
        }
    }
    
    set.seed(1234)
    sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution/100)
    
    write.csv(sp_data@reductions$pca@cell.embeddings, file = file.path(output, 'Seurat_final.csv'), quote=F)
    write.csv(sp_data@meta.data, file = file.path(output, 'Seurat.csv'), quote=FALSE)
}


show_seurat(dataroot, file_name, n_clusters, savepath)

