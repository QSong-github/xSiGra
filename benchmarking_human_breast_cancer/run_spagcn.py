import argparse
import os
import random

import anndata
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
import torch
import matplotlib.pyplot as plt

def show_spagcn(adata, img, ncluster):
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata = adata

    # Normalize
    adata.var_names_make_unique()
    sc.pp.normalize_per_cell(adata)

    arr = adata.obsm["spatial"]

    # Create a DataFrame from arr
    df = pd.DataFrame(arr, index=adata.obs.index, columns=["cx", "cy"])

    adata.obs["cx"] = df["cx"].astype(np.int64)
    adata.obs["cy"] = df["cy"].astype(np.int64)
    adata.obs["x2"] = df["cy"].astype(np.int64)
    adata.obs["x3"] = df["cx"].astype(np.int64)
    adata.obs["x4"] = df["cy"].astype(np.int64)
    adata.obs["x5"] = df["cx"].astype(np.int64)

    # Single cell data has no spots hence using pixel for both

    s = 1
    b = 49
    x_array = adata.obs["x2"].tolist()
    y_array = adata.obs["x3"].tolist()
    x_pixel = adata.obs["x4"].tolist()
    y_pixel = adata.obs["x5"].tolist()

    # Calculate adjacent matrix
    adj = spg.calculate_adj_matrix(
        x=x_array,
        y=y_array,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        image=img,
        beta=b,
        alpha=s,
        histology=True,
    )
    sc.pp.log1p(adata)

    p = 0.5

    # Find the l value given p
    l = spg.search_l(p, adj, start=0.1, end=1000, tol=0.01, max_run=100)

    r_seed = t_seed = n_seed = 100
    res = spg.search_res(
        adata,
        adj,
        l,
        ncluster,
        start=0.6,
        step=0.1,
        tol=5e-4,
        lr=0.001,
        max_epochs=1000,
        r_seed=r_seed,
        t_seed=t_seed,
        n_seed=n_seed,
    )

    # Run SpaGCN
    clf = spg.SpaGCN()
    clf.set_l(l)

    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=1000,
    )
    adata.obsm["embed"] = clf.embed
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype("category")
    print(adata.obs["pred"].unique())
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = spg.refine(
        sample_id=adata.obs.index.tolist(),
        pred=adata.obs["pred"].tolist(),
        dis=adj_2d,
        shape="hexagon",
    )
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype("category")
    
    obs_df = adata.obs.dropna()
    adata = adata[obs_df.index, :]

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    silhouette = silhouette_score(adata.obsm["embed"], obs_df["pred"])
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(adata.obsm["embed"], obs_df["pred"])
    print('Davies-Bouldin Score: %.2f' % davies_bouldin)

    calinski = calinski_harabasz_score(adata.obsm["embed"], obs_df['pred'])
    print('Calinski Score: %.2f' % calinski)

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../dataset/breast_invasive_carcinoma/")
    parser.add_argument("--save_path", type=str, default="./HBC_SpaGCN")
    parser.add_argument("--ncluster", type=int, default=8)
    opt = parser.parse_args()

    root = opt.path
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read data
    adata = sc.read_visium(root, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    adata.X = adata.X.A

    image = cv2.imread(os.path.join(root, "Visium_FFPE_Human_Breast_Cancer_image.tif"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    ncluster = opt.ncluster
    adata.X = adata.X.astype('float64')
    
    # Run SpaGCN
    adata = show_spagcn(adata, image, ncluster)

    # Spatial plot
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)
    ax=sc.pl.spatial(adata, color=['pred'], title=['SpaGCN'], show=False)
    plt.savefig(os.path.join(save_path, 'spagcn_spatial.pdf'), bbox_inches='tight')
