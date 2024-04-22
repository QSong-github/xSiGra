args = commandArgs(trailingOnly=TRUE)
dataroot <- args[1]
num_fovs <- args[2]
n_clusters <- args[3]
savepath <- args[4]
sample.name <- args[5]

library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(ggplot2)
library(patchwork)
library(dplyr)
options(bitmapType = 'cairo')
library(anndata)

show_seurat <- function(input_root, num_fovs, n_cluster, savepath, sample.name){
    for (i in 1:num_fovs){
        fovid = paste0('fov',i)
        print(fovid)
        
        input = file.path(input_root, fovid, 'sampledata.h5seurat')
        output = file.path(savepath, sample.name)
        if (!dir.exists(file.path(output))){
            dir.create(file.path(output), recursive = TRUE)
        }

        # Read Seurat object
        sp_data = LoadH5Seurat(input, meta.data = FALSE)
        
        # Remove Nan
        sp_data = na.omit(sp_data)

	ab <- colSums(sp_data)
        sp_data@meta.data$sums <- ab
        sp_data <- subset(sp_data,sums!=0)
        
        print('SCTransofrom')
        set.seed(1234)

        # SCTransform normalization
        sp_data <- SCTransform(sp_data, assay = "RNA", verbose = T, variable.features.n = 980)
        set.seed(1234)
        
        # Run PCA
        sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
        sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:50)
        for(resolution in 5:50){
            set.seed(1234)

            sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
            
            if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
                break
            }
        }
        set.seed(1234)
        
        # Clustering
        sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution/100)
        
        saveRDS(sp_data, file.path(output, paste(fovid,'_Seurat_final.rds',sep='')))
        write.table(sp_data@reductions$pca@cell.embeddings, file = file.path(output, paste(fovid,'_seurat.PCs.csv',sep='')), sep='\t', quote=F)
        
        # For each fov save the cluster results
        write.table(sp_data@meta.data, file = file.path(output, paste(fovid,'.csv',sep='')), sep='\t', quote=FALSE)
    }
}
show_seurat(dataroot, num_fovs, n_clusters, savepath, sample.name)
