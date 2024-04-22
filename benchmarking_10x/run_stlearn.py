import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics import (
    adjusted_rand_score,
)


def run_stlearn(
    sample="151676",
    imname="tissue_hires_image.png",
    BASE_PATH="../dataset/10x/",
    name="sampledata.h5ad",
    save_path="./10x_stlearn/151676/",
    n_cluster=7,
):
    # Set paths
    TILE_PATH = Path("/tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("%s/%s/%s" % (save_path, sample, "sample"))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Read data
    data = sc.read(os.path.join(BASE_PATH, sample, name))

    Ann_df = pd.read_csv(
        "%s/%s/annotation.txt" % (opt.root, opt.sample),
        sep="\t",
        header=None,
        index_col=0,
    )
    Ann_df.columns = ["Ground Truth"]
    data.obs["Ground Truth"] = Ann_df.loc[data.obs_names, "Ground Truth"]

    if ".npy" in imname:
        img = np.load(os.path.join(BASE_PATH, sample, "spatial", imname))
        img = img[:, :, :3]
    else:
        img = cv2.imread(os.path.join(BASE_PATH, sample, "spatial", imname))

    coors = data.obsm["spatial"]
    data.obs["px"] = coors[:, 0]
    data.obs["py"] = coors[:, 1]

    data.uns["spatial"] = {id: {"images": {"hires": img / 255.0}, "use_quality": "hires"}}
    data.obs["imagerow"] = data.obs["px"]
    data.obs["imagecol"] = data.obs["py"]

    # Pre-process
    st.pp.filter_genes(data, min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)
    st.em.run_pca(data, n_comps=15)
    st.pp.tiling(data, TILE_PATH)
    st.pp.extract_feature(data)

    # SME normalization
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm["raw_SME_normalized"]
    st.pp.scale(data_)

    # PCA
    st.em.run_pca(data_, n_comps=30)
    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")

    obs_df = data_.obs.dropna()

    # Compute ARI
    ARI = adjusted_rand_score(obs_df["X_pca_kmeans"], obs_df["Ground Truth"])
    ARI = round(ARI, 2)
    print(ARI)
    labels = obs_df["Ground Truth"]

    # Read data
    adata = sc.read(os.path.join(BASE_PATH, sample, name))

    # Pre-process
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.pca(adata, svd_solver="arpack")
    adata.obs["Ground Truth"] = data.obs["Ground Truth"]
    adata.obs["leiden"] = data_.obs["X_pca_kmeans"]
    obs_df = adata.obs.dropna()

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print(ARI)

    labels = obs_df["Ground Truth"]

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

    # Spatial figure
    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Slice 151674"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial.pdf"), bbox_inches="tight")

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["stLearn (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial.pdf"), bbox_inches="tight")

    # Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="stLearn", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final.pdf"), bbox_inches="tight")

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print("ARI: %.2f" % ARI)

    return data_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/10x/")
    parser.add_argument("--imgname", type=str, default="tissue_hires_image.png")
    parser.add_argument("--num_cluster", type=int, default=7)
    parser.add_argument("--save_path", type=str, default="./10x_stlearn/151676/")
    parser.add_argument("--save_name", type=str, default="stlearn.csv")
    parser.add_argument("--sample", type=str, default="151676")
    opt = parser.parse_args()

    sample = opt.sample
    imname = opt.imgname
    num_cluster = opt.num_cluster
    name = "sampledata.h5ad"
    data = run_stlearn(
        sample=sample,
        imname=imname,
        BASE_PATH=opt.root,
        name=name,
        save_path=opt.save_path,
        n_cluster=num_cluster,
    )
    save_path = os.path.join(opt.save_path, opt.save_name)

    # Save results to csv
    df = data.obs
    df.to_csv(save_path)
