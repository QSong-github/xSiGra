import argparse
import os
import random

import matplotlib.pyplot as plt
import cv2
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

def show_scanpy(adata, ncluster):

    # Preprocess
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=4,use_rep="X", method="rapids")

    # Apply pca
    sc.tl.pca(adata, n_comps=150, svd_solver="arpack")

    def res_search(adata_pred, ncluster, seed, iter=200):
        start = 0
        end = 3
        i = 0
        while start < end:
            if i >= iter:
                return res
            i += 1
            res = (start + end) / 2
            print(res)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
            count = len(set(adata_pred.obs["leiden"]))
            if count == ncluster:
                print("find", res)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()

    # Perform clustering
    res = res_search(adata, ncluster, 1234)
    sc.tl.leiden(adata, resolution=res, random_state=1234)
    print(len(set(list(adata.obs["leiden"]))))

    obs_df = adata.obs.dropna()
    adata = adata[obs_df.index, :]
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    silhouette = silhouette_score(adata.X, obs_df['leiden'])
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(adata.X, obs_df['leiden'])
    print('Davies-Bouldin Score: %.2f' % davies_bouldin)

    calinski = calinski_harabasz_score(adata.X, obs_df['leiden'])
    print('Calinski Score: %.2f' % calinski)

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../dataset/breast_invasive_carcinoma/")
    parser.add_argument("--save_path", type=str, default="./HBC_scanpy")
    parser.add_argument("--ncluster", type=int, default=8)

    opt = parser.parse_args()

    
    path = opt.path
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Read data
    adata = sc.read_visium(path, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    
    adata.var_names_make_unique()
    adata.X = adata.X.A

    ncluster = opt.ncluster
    adata = show_scanpy(adata, ncluster)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)
    ax=sc.pl.spatial(adata, color=['leiden'], title=['Scanpy'], show=False)
    plt.savefig(os.path.join(save_path, 'scanpy_spatial.pdf'), bbox_inches='tight')