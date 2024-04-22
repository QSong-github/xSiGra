import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st
from sklearn.metrics import (
    adjusted_rand_score,
)
from sklearn.preprocessing import LabelEncoder


def run_stlearn(sample, imname, BASE_PATH, save_path, ncluster):
    TILE_PATH = Path("/tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("%s/%s" % (save_path, sample))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Read data
    data = sc.read(os.path.join(BASE_PATH, sample, "sampledata.h5ad"))
    ground_truth_df = pd.DataFrame()
    ground_truth_df["ground_truth"] = data.obs["merge_cell_type"]

    # Read image
    img = cv2.imread(os.path.join(BASE_PATH, sample, "CellComposite_%s.jpg" % (imname)))
    data.uns["spatial"] = {id: {"images": {"hires": img / 255.0}, "use_quality": "hires"}}
    data.obs["imagerow"] = data.obs["cx"]
    data.obs["imagecol"] = data.obs["cy"]
    data.obs["array_row"] = data.obs["cx"]
    data.obs["array_col"] = data.obs["cy"]

    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))

    n_cluster = ncluster
    data.obs["ground_truth"] = data.obs["merge_cell_type"]

    ground_truth_df["ground_truth_le"] = ground_truth_le

    # Pre-process
    st.pp.filter_genes(data, min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)
    st.em.run_pca(data, n_comps=15)
    st.pp.tiling(data, TILE_PATH)
    st.pp.extract_feature(data)

    # Run SME learning
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm["raw_SME_normalized"]
    st.pp.scale(data_)

    # Run PCA
    st.em.run_pca(data_, n_comps=30)

    # Kmeans clustering
    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")

    df = data_.obs.dropna()

    # Compute ARI
    ari = adjusted_rand_score(df["X_pca_kmeans"], df["ground_truth"])
    ari = round(ari, 2)
    print(ari)
    return ari, data_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/nanostring/")
    parser.add_argument("--save_path", type=str, default="./stlearn/")
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--num_fov", type=int, default=20)
    opt = parser.parse_args()

    BASE_PATH = Path(opt.root)
    samples = ["fov%d" % (i) for i in range(1, opt.num_fov + 1)]
    imnames = ["F0%02d" % (i) for i in range(1, opt.num_fov + 1)]

    for s, imname in zip(samples, imnames):
        # Run stlearn and compute ARI
        ari, adata = run_stlearn(s, imname, BASE_PATH, opt.save_path, opt.ncluster)
        print(ari)

        ind = adata.obs["X_pca_kmeans"].isna()
        adata = adata[~ind]

        df = pd.DataFrame(adata.obsm["raw_SME_normalized"], index=adata.obs.index)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)

        df.to_csv(os.path.join(opt.save_path, s, "%s_emb.csv" % s))

        # Save stlearn clusters
        df = pd.DataFrame(index=adata.obs.index)
        df["cluster"] = adata.obs["X_pca_kmeans"]
        df["merge_cell_type"] = adata.obs["merge_cell_type"]
        df.to_csv(os.path.join(opt.save_path, s, "%s_cluster.csv" % s))
