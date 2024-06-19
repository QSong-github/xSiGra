import os
import cv2
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def get_ari_bayesspace(root='bayesspace', key='spatial.cluster', name="HBC"):

    # Read embeddings and cluster labels
    csv = pd.read_csv(os.path.join(root, 'bayesSpace.csv'), header=0, index_col=0)
    embeddings = pd.read_csv(os.path.join(root, 'bayesSpace_PCA_embeddings.csv'), header=0, index_col=0, sep="\t")
    pred_all = csv[key]

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    silhouette = silhouette_score(embeddings, pred_all)
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(embeddings, pred_all)
    print('Davies-Bouldin Score: %.2f\n' % davies_bouldin)

    calinski = calinski_harabasz_score(embeddings, pred_all)
    print('Calinski Score: %.2f' % calinski)

    adata.obs["pred"] = pred_all
    adata.obs["pred"] = adata.obs["pred"].astype(str)

    # Spatial plot
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = root
   
    ax=sc.pl.spatial(adata, color=['pred'], title=['BayesSpace'], show=False)
    plt.savefig(os.path.join(root, 'bayesspace_spatial.pdf'), bbox_inches='tight')
    
def get_ari_seurat(root='seurat', key='seurat_clusters', sep=',', dataroot='spagcn/lung5-1', name="HBC"):
    
    # Read embeddings and cluster labels
    csv = pd.read_csv(os.path.join(root, 'Seurat.csv'), header=0, index_col=0, sep=sep)  
    embeddings = pd.read_csv(os.path.join(root, 'Seurat_final.csv'), header=0, index_col=0, sep=sep)
    pred_all = csv[key]
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    silhouette = silhouette_score(embeddings, pred_all)
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(embeddings, pred_all)
    print('Davies-Bouldin Score: %.2f\n' % davies_bouldin)

    calinski = calinski_harabasz_score(embeddings, pred_all)
    print('Calinski Score: %.2f' % calinski)

    adata.obs["pred"] = pred_all
    adata.obs["pred"] = adata.obs["pred"].astype(str)

    # Spatial plot
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = root
   
    ax=sc.pl.spatial(adata, color=['pred'], title=['Seurat'], show=False)
    plt.savefig(os.path.join(root, 'seurat_spatial.pdf'), bbox_inches='tight')

# Load data
path = "../dataset/breast_invasive_carcinoma"
adata = sc.read_visium(path, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")    
adata.var_names_make_unique()
adata.X = adata.X.A

get_ari_seurat(key='seurat_clusters', root='./HBC_Seurat/', name="HBC")
get_ari_bayesspace(root='./HBC_BayesSpace', name="HBC")