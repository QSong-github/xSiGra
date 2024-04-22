import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score


def show_scanpy(opt, root, save_path, num_cluster=7, seed=1234):
    # Read data
    adata = sc.read(root)

    # Preprocess
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # Perform PCA
    sc.tl.pca(adata, svd_solver="arpack")

    # Read ground truth
    Ann_df = pd.read_csv(
        "%s/annotation.txt" % ("/".join(opt.root.split("/")[:4])),
        sep="\t",
        header=None,
        index_col=0,
    )
    Ann_df.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]

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
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
            count = len(set(adata_pred.obs["leiden"]))

            if count == int(ncluster):
                print("find", res)
                return res
            if count > int(ncluster):
                end = res
            else:
                start = res
        raise NotImplementedError()

    # Clustering
    res = res_search(adata, num_cluster, seed)
    sc.tl.leiden(adata, resolution=res, random_state=seed)
    print(len(set(list(adata.obs["leiden"]))))

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print(ARI)

    labels = obs_df["Ground Truth"]
    print(len(obs_df["Ground Truth"]))
    print(len(obs_df["leiden"]))

    # Match clusters to cell types
    def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
        num_samples = flat_target.shape[0]
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k))
        for c1 in range(num_k):
            for c2 in range(num_k):
                votes = int(((flat_preds == c1) * (flat_target == c2)).sum())
                num_correct[c1, c2] = votes
        match = linear_sum_assignment(num_samples - num_correct)
        match = np.array(list(zip(*match)))
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))
        return res

    le = preprocessing.LabelEncoder()
    gt = le.fit_transform(labels)

    match = _hungarian_match(obs_df["leiden"].astype(np.int8), gt.astype(np.int8), num_cluster, num_cluster)
    dict1 = {}
    dict_name = {}
    for i in match:
        dict1[str(i[0])] = i[1]
        dict_name[i[1]] = le.classes_[i[0]]

    obs_df = obs_df.replace({"leiden": dict1})

    adata.obs["leiden"] = obs_df["leiden"]

    plt.rcParams["figure.figsize"] = (3, 3)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path

    # Spatial figures
    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Ground Truth"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial1.pdf"), bbox_inches="tight")

    print(obs_df["Ground Truth"])

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["Scanpy (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial1.pdf"), bbox_inches="tight")

    # Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="Scanpy", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final2.pdf"), bbox_inches="tight")

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print("ARI: %.2f" % ARI)

    return adata, ARI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/10x/151676/sampledata.h5ad")
    parser.add_argument("--num_cluster", type=int, default=7)
    parser.add_argument("--save_path", type=str, default="./10x_scanpy/151676")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_name", type=str, default="scanpy.h5ad")
    opt = parser.parse_args()

    root = opt.root
    num_cluster = opt.num_cluster
    save_path = os.path.join(opt.save_path)
    adata, ARI = show_scanpy(opt, root, save_path, num_cluster, seed=opt.seed)
