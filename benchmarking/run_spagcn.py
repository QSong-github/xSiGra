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
from sklearn.metrics.cluster import adjusted_rand_score


def show_spagcn(root, id, imgname, ncluster):
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load data
    dataroot = os.path.join(root, id, "sampledata.h5ad")
    adata = sc.read(dataroot)

    # Normalize
    adata.var_names_make_unique()
    sc.pp.normalize_per_cell(adata)

    imgroot = os.path.join(root, id, "CellComposite_%s.jpg" % (imgname))
    img = cv2.imread(imgroot)

    adata.obs["x2"] = adata.obs["cy"].astype(np.int64)
    adata.obs["x3"] = adata.obs["cx"].astype(np.int64)
    adata.obs["x4"] = adata.obs["cy"].astype(np.int64)
    adata.obs["x5"] = adata.obs["cx"].astype(np.int64)

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
        start=1.6,
        step=0.1,
        tol=5e-4,
        lr=0.001,
        max_epochs=500,
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
        max_epochs=200,
    )
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype("category")

    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = spg.refine(
        sample_id=adata.obs.index.tolist(),
        pred=adata.obs["pred"].tolist(),
        dis=adj_2d,
        shape="hexagon",
    )
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype("category")

    gt = adata.obs["merge_cell_type"].astype("category").cat.codes
    cellid2name = {}
    for gt, name in zip(list(gt), adata.obs["merge_cell_type"]):
        if not gt in cellid2name:
            cellid2name[gt] = name

    # Compute ARI
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df["refined_pred"], obs_df["merge_cell_type"])
    print("ari is %.2f" % (ARI))

    return ARI, adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/nanostring/lung13")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_path", type=str, default="../checkpoint/nanostring/lung13")
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--num_fov", type=int, default=20)
    opt = parser.parse_args()

    num_fovs = opt.num_fov
    ids = ["fov%d" % (i) for i in range(1, num_fovs + 1)]
    imgnames = ["F0%02d" % (i) for i in range(1, num_fovs + 1)]

    repeat_time = 1
    keep_record = dict()

    root = opt.root
    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    display_results = {}

    adatas = []

    # For each fov save cluster results
    for id, imname in zip(ids, imgnames):
        print(id, imname)
        ncluster = opt.ncluster
        display_results[id] = []
        ARI, adata = show_spagcn(root, id, imname, ncluster)
        display_results[id].append([ARI])
        refine_pred = adata.obs["refined_pred"]
        merge_cell = adata.obs["merge_cell_type"]
        df = pd.DataFrame(index=adata.obs.index)
        df["refined_pred"] = refine_pred
        df["merge_cell_type"] = merge_cell

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(os.path.join(save_path, "%s.csv" % (id)))
        adatas.append(adata)

    arrays = []
    for k, v in display_results.items():
        arrays.append(v[0])

    arr = np.array(arrays)
    df = pd.DataFrame(arr, columns=["ari"], index=ids)
    df.to_csv(os.path.join(save_path, "spagcn.csv"))

    adata_pred = anndata.concat(adatas)
    adata_pred.write(os.path.join(save_path, "spagcn.h5ad"))
