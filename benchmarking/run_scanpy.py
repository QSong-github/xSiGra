import argparse
import os
import random

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score


def show_scanpy(root, id, ncluster):
    # Read data
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))

    # Preprocess
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var["highly_variable"]]
    else:
        adata = adata
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep="X", method="rapids")

    # Apply pca
    sc.tl.pca(adata, svd_solver="arpack")

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

    # Compute ARI
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
    print(ARI)
    return ARI, adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/nanostring/lung9-rep1")
    parser.add_argument("--save_path", type=str, default="./scanpy/")
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--num_fov", type=int, default=20)
    opt = parser.parse_args()

    root = opt.root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_fov = opt.num_fov

    ids = ["fov%d" % i for i in range(1, num_fov + 1)]
    ncluster = opt.ncluster

    keep_record = dict()

    display_results = {}
    adatas = []

    # For each fov compute and save cell clusters
    for it, id in enumerate(ids):
        display_results[id] = []
        ARI, adata = show_scanpy(root, id, ncluster)
        display_results[id].append([ARI])
        adatas.append(adata)

        df = pd.DataFrame(index=adata.obs.index)
        df["scanpy"] = adata.obs["leiden"]
        df["merge_cell_type"] = adata.obs["merge_cell_type"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(os.path.join(save_path, "%s.csv" % id))

    arrays = []
    for k, v in display_results.items():
        arrays.append(v[0])

    arr = np.array(arrays)

    # Save to csv
    df = pd.DataFrame(arr, columns=["ari"], index=ids)
    df.to_csv(os.path.join(save_path, "scanpy.csv"))

    adata_pred = anndata.concat(adatas)
    adata_pred.write(os.path.join(save_path, "scanpy.h5ad"))
