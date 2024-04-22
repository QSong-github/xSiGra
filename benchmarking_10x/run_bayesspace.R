args = commandArgs(trailingOnly=TRUE)
dataroot <- args[1]
n_clusters <- as.integer(args[2])
savepath <- args[3]
h5_name <- args[4]

library(BayesSpace)
library(ggplot2)
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(anndata)
library(dplyr)

show_BayesSpace <- function(input_root, n_cluster, savepath, h5_name){
    dir.input = input_root
    output = savepath
    if (!dir.exists(file.path(output))){
        dir.create(file.path(output), recursive = TRUE)
    }
    print(dir.input)
    print(h5_name)

    # Load 10x data
    sp_data <- Load10X_Spatial(dir.input, filename = h5_name)

    # Read metadata
    df_meta <- read.csv(file.path(dir.input, 'spatial/tissue_positions_list.csv'),
                        stringsAsFactors = F, header=F, check.names=F, row.names=1)

    df_meta1 = df_meta[match(colnames(sp_data),rownames(df_meta)),]
    df_meta1 = df_meta1[,-1,drop=FALSE]
    colnames(df_meta1) = c('sx','sy','px','py')
    identical(rownames(df_meta1),colnames(sp_data))
    sp_data@meta.data = cbind(sp_data@meta.data, df_meta1)
    
    dlpfc = sp_data

    ab <- colSums(dlpfc)
    library(rhdf5)
    
    foo1 <- dlpfc@meta.data
    dlpfc@meta.data = foo1[,c('sx','sy','px','py')]
    X <- dlpfc[['Spatial']]@counts

    # Construct single cell experiment object
    dlpfc <- SingleCellExperiment(X,colData = dlpfc@meta.data)
    libsizes <- colSums(X)
    size.factors <- libsizes/mean(libsizes)
    logcounts(dlpfc) <- log2(t(t(X)/size.factors) + 1)

    set.seed(1234)
    dec <- scran::modelGeneVar(dlpfc)
    top <- scran::getTopHVGs(dec, n = 500)

    set.seed(1234)
    dlpfc <- scater::runPCA(dlpfc, subset_row=top)

    # Add BayesSpace metadata
    dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)
        
    # Clustering with BayesSpace
    q <- n_clusters  # Number of clusters
    d <- 15  # Number of PCs

    # Run BayesSpace clustering
    set.seed(1234)
    colnames(colData(dlpfc))[1:2] = c('row','col')
    
    dlpfc <- spatialCluster(dlpfc, q=q, d=d, nrep=5000, gamma=3, platform="Visium", save.chain=FALSE)
    
    labels <- dlpfc$spatial.cluster
    label <- data.frame(pred=labels)
    gp <- dlpfc$layer_guess

    # Save to csv
    write.csv(label,file=file.path(output,'bayesSpace.csv'), quote=F)
    write.csv(colData(dlpfc),file=file.path(output,'bayesSpace.csv'), quote=FALSE)
}

show_BayesSpace(dataroot, n_clusters, savepath,h5_name)