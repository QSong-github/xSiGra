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


def get_ari(csv_path, key, root, save_path, num_cluster=4, seed=1234):
    # Read data
    adata = sc.read(root)

    # Preprocess
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.pca(adata, svd_solver="arpack")

    # Read annotation
    Ann_df = pd.read_csv(
        "%s/annotation.txt" % ("/".join(root.split("/")[:4])),
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

            if count == ncluster:
                print("find", res)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()

    res = res_search(adata, num_cluster, seed)
    sc.tl.leiden(adata, resolution=res, random_state=seed)

    # Read saved csv and use clusters
    csv_data = pd.read_csv(csv_path)
    leiden = csv_data[key].tolist()

    leiden = list(map(int, leiden))
    adata.obs["leiden"] = leiden
    obs_df = adata.obs.dropna()

    # Compute ARI
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print(ARI)

    labels = obs_df["Ground Truth"]

    # Match cluster to cell type
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
        dict1[int(i[0])] = int(i[1])
        dict_name[i[1]] = le.classes_[i[0]]

    # Replace matched leiden with ground truth
    obs_df = obs_df.replace({"leiden": dict1})

    adata.obs["leiden"] = obs_df["leiden"].astype(np.int8)

    plt.rcParams["figure.figsize"] = (3, 3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Draw spatial figures
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path

    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Ground Truth"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial.pdf"), bbox_inches="tight")

    df = adata.obs["leiden"]

    df = df.fillna(7)
    leiden = df.tolist()
    leiden = [int(x) for x in leiden]
    leiden = [str(x) for x in leiden]
    adata.obs["leiden"] = leiden

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["Seurat (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial.pdf"), bbox_inches="tight")

    # Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="Seurat", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final.pdf"), bbox_inches="tight")

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print("ARI: %.2f" % ARI)


def get_ari_bayesspace(csv_path, key, root, save_path, num_cluster=4, seed=1234):
    # Read anndata
    adata = sc.read(root)

    # Preprocess
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.pca(adata, svd_solver="arpack")

    # Read annotation
    Ann_df = pd.read_csv(
        "%s/annotation.txt" % ("/".join(root.split("/")[:4])),
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

            if count == ncluster:
                print("find", res)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()

    res = res_search(adata, num_cluster, seed)

    # Read saved csv and use clusters
    csv_data = pd.read_csv(csv_path)

    leiden = csv_data[key].tolist()
    dict1 = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
    }
    leiden = [dict1[x] for x in leiden]

    leiden = list(map(int, leiden))
    adata.obs["leiden"] = leiden
    obs_df = adata.obs.dropna()

    # Compute ARI
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print(ARI)

    labels = obs_df["Ground Truth"]

    # Match cluster to cell type
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
        dict1[int(i[0])] = int(i[1])
        dict_name[i[1]] = le.classes_[i[0]]

    # Replace matched leiden with ground truth
    obs_df = obs_df.replace({"leiden": dict1})

    adata.obs["leiden"] = obs_df["leiden"].astype(np.int8)

    plt.rcParams["figure.figsize"] = (3, 3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Draw spatial figures
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path

    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Ground Truth"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial.pdf"), bbox_inches="tight")

    df = adata.obs["leiden"]

    df = df.fillna(7)
    leiden = df.tolist()
    leiden = [int(x) for x in leiden]
    leiden = [str(x) for x in leiden]
    adata.obs["leiden"] = leiden

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["Bayesspace (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial.pdf"), bbox_inches="tight")

    # UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="Bayesspace", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final.pdf"), bbox_inches="tight")

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print("ARI: %.2f" % ARI)


# Stagate
get_ari(
    csv_path="./10x_stagate/151676/151676_STAGATE.csv",
    key="leiden",
    root="../dataset/10x/151676/sampledata.h5ad",
    save_path="./10x_stagate/151676/",
    num_cluster=7,
    seed=1234,
)
# get_ari(
#     csv_path="./10x_stagate/151675/151675_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151675/sampledata.h5ad",
#     save_path="./10x_stagate/151675/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151674/151674_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151674/sampledata.h5ad",
#     save_path="./10x_stagate/151674/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151673/151673_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151673/sampledata.h5ad",
#     save_path="./10x_stagate/151673/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151672/151672_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151672/sampledata.h5ad",
#     save_path="./10x_stagate/151672/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151671/151671_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151671/sampledata.h5ad",
#     save_path="./10x_stagate/151671/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151670/151670_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151670/sampledata.h5ad",
#     save_path="./10x_stagate/151670/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151669/151669_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151669/sampledata.h5ad",
#     save_path="./10x_stagate/151669/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151510/151510_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151510/sampledata.h5ad",
#     save_path="./10x_stagate/151510/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151509/151509_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151509/sampledata.h5ad",
#     save_path="./10x_stagate/151509/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151508/151508_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151508/sampledata.h5ad",
#     save_path="./10x_stagate/151508/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stagate/151507/151507_STAGATE.csv",
#     key="leiden",
#     root="../dataset/10x/151507/sampledata.h5ad",
#     save_path="./10x_stagate/151507/",
#     num_cluster=7,
#     seed=1234,
# )

# stlearn
get_ari(
    csv_path="./10x_stlearn/151676/stlearn.csv",
    key="X_pca_kmeans",
    root="../dataset/10x/151676/sampledata.h5ad",
    save_path="./10x_stlearn/151676/",
    num_cluster=7,
    seed=1234,
)
# get_ari(
#     csv_path="./10x_stlearn/151675/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151675/sampledata.h5ad",
#     save_path="./10x_stlearn/151675/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151674/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151674/sampledata.h5ad",
#     save_path="./10x_stlearn/151674/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151673/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151673/sampledata.h5ad",
#     save_path="./10x_stlearn/151673/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151672/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151672/sampledata.h5ad",
#     save_path="./10x_stlearn/151672/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151671/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151671/sampledata.h5ad",
#     save_path="./10x_stlearn/151671/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151670/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151670/sampledata.h5ad",
#     save_path="./10x_stlearn/151670/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151669/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151669/sampledata.h5ad",
#     save_path="./10x_stlearn/151669/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151510/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151510/sampledata.h5ad",
#     save_path="./10x_stlearn/151510/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151509/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151509/sampledata.h5ad",
#     save_path="./10x_stlearn/151509/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151508/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151508/sampledata.h5ad",
#     save_path="./10x_stlearn/151508/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_stlearn/151507/stlearn.csv",
#     key="X_pca_kmeans",
#     root="../dataset/10x/151507/sampledata.h5ad",
#     save_path="./10x_stlearn/151507/",
#     num_cluster=7,
#     seed=1234,
# )

get_ari(
    csv_path="./10x_seurat/151676/Seurat.csv",
    key="seurat_clusters",
    root="../dataset/10x/151676/sampledata.h5ad",
    save_path="./10x_seurat/151676/",
    num_cluster=7,
    seed=1234,
)
# get_ari(
#     csv_path="./10x_seurat/151675/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151675/sampledata.h5ad",
#     save_path="./10x_seurat/151675/",
#     num_cluster=7,
#     seed=1234,
# )

# # Seurat
# get_ari(
#     csv_path="./10x_seurat/151674/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151674/sampledata.h5ad",
#     save_path="./10x_seurat/151674/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151673/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151673/sampledata.h5ad",
#     save_path="./10x_seurat/151673/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151672/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151672/sampledata.h5ad",
#     save_path="./10x_seurat/151672/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151671/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151671/sampledata.h5ad",
#     save_path="./10x_seurat/151671/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151670/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151670/sampledata.h5ad",
#     save_path="./10x_seurat/151670/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151669/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151669/sampledata.h5ad",
#     save_path="./10x_seurat/151669/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151510/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151510/sampledata.h5ad",
#     save_path="./10x_seurat/151510/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151509/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151509/sampledata.h5ad",
#     save_path="./10x_seurat/151509/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151508/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151508/sampledata.h5ad",
#     save_path="./10x_seurat/151508/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari(
#     csv_path="./10x_seurat/151507/Seurat.csv",
#     key="seurat_clusters",
#     root="../dataset/10x/151507/sampledata.h5ad",
#     save_path="./10x_seurat/151507/",
#     num_cluster=7,
#     seed=1234,
# )

# BayesSpace
get_ari_bayesspace(
    csv_path="./10x_bayesspace/151676/bayesSpace.csv",
    key="spatial.cluster",
    root="../dataset/10x/151676/sampledata.h5ad",
    save_path="./10x_bayesspace/151676/",
    num_cluster=7,
    seed=1234,
)
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151675/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151675/sampledata.h5ad",
#     save_path="./10x_bayesspace/151675/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151674/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151674/sampledata.h5ad",
#     save_path="./10x_bayesspace/151674/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151673/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151673/sampledata.h5ad",
#     save_path="./10x_bayesspace/151673/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151672/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151672/sampledata.h5ad",
#     save_path="./10x_bayesspace/151672/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151671/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151671/sampledata.h5ad",
#     save_path="./10x_bayesspace/151671/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151670/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151670/sampledata.h5ad",
#     save_path="./10x_bayesspace/151670/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151669/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151669/sampledata.h5ad",
#     save_path="./10x_bayesspace/151669/",
#     num_cluster=5,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151510/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151510/sampledata.h5ad",
#     save_path="./10x_bayesspace/151510/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151509/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151509/sampledata.h5ad",
#     save_path="./10x_bayesspace/151509/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151508/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151508/sampledata.h5ad",
#     save_path="./10x_bayesspace/151508/",
#     num_cluster=7,
#     seed=1234,
# )
# get_ari_bayesspace(
#     csv_path="./10x_bayesspace/151507/bayesSpace.csv",
#     key="spatial.cluster",
#     root="../dataset/10x/151507/sampledata.h5ad",
#     save_path="./10x_bayesspace/151507/",
#     num_cluster=7,
#     seed=1234,
# )
