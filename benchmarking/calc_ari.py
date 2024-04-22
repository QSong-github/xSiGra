import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score


# Match clusters to cell types
def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_preds.shape[0]
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


def match_cluster_to_cell(df_all, key):
    leiden_number = list(set(df_all[key]))
    dicts = {}
    for ln in leiden_number:
        ind = df_all[key] == ln
        temp = df_all[ind]
        df = temp["merge_cell_type"].value_counts()
        dicts[int(ln)] = df.index[0]
    return dicts


def get_stlearn(root="stlearn", num_fovs=30, key="cluster"):
    pred_all = []
    gt_all = []
    index = []
    for i in range(1, num_fovs + 1):
        # Read saved csv files with cluster and celltype for each fov
        csv = pd.read_csv(
            os.path.join(root, "fov%d" % i, "fov%d_cluster.csv" % (i)),
            header=0,
            index_col=0,
        )
        index.extend(list(csv.index))
        pred = csv[key].astype("category").cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred
        gt = csv["merge_cell_type"].astype("category")
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)

        id2cell = {}
        for g, gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g

        match = _hungarian_match(pred, gt_cat, len(set(pred)), len(set(gt_cat)))

        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc

        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)

    # Compute ARI and save results to csv
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df["pred"] = pred_all
    df["gt"] = gt_all
    df.to_csv("./cluster_results_stlearn/stlearn.csv")
    print("%s: ari:%.2f" % (root, ari))


def get_bayesspace(root="bayesspace", num_fovs=30, key="pred"):
    pred_all = []
    gt_all = []
    index = []
    for i in range(1, num_fovs + 1):
        # Read csv for each fov
        csv = pd.read_csv(os.path.join(root, "fov%d_bayesSpace.csv" % (i)), header=0, index_col=0)
        index.extend(list(csv.index))
        pred = csv[key].astype("category").cat.codes
        pred = csv[key].astype(np.int8)
        csv["merge_cell_type"] = csv["gt"]
        csv[key] = pred
        gt = csv["merge_cell_type"].astype("category")
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)

        id2cell = {}
        for g, gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g

        match = _hungarian_match(pred, gt_cat, len(set(pred)), len(set(gt_cat)))

        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc

        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)

    # Compute ARI and save results to csv
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df["pred"] = pred_all
    df["gt"] = gt_all
    df.to_csv("./cluster_results_bayesspace/bayesspace.csv")
    print("%s: ari:%.2f" % (root, ari))


def get_seurat(
    root="seurat",
    num_fovs=30,
    key="seurat_clusters",
    sep="\t",
    dataroot="spagcn/lung5-1",
):
    pred_all = []
    gt_all = []
    index = []

    for i in range(1, num_fovs + 1):
        # Read csv for each fov
        csv = pd.read_csv(os.path.join(root, "fov%d.csv" % (i)), header=0, index_col=0, sep=sep)

        index.extend(list(csv.index))
        adata = pd.read_csv(os.path.join(dataroot, "fov%d.csv" % (i)), header=0, index_col=0)

        csv["merge_cell_type"] = adata.loc[csv.index, "merge_cell_type"]
        pred = csv[key].astype("category").cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred
        gt = csv["merge_cell_type"].astype("category")
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)

        id2cell = {}
        for g, gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g

        match = _hungarian_match(pred, gt_cat, len(set(pred)), len(set(gt_cat)))

        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc

        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)

    # Cimoute ARI and save results to csv
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df["pred"] = pred_all
    df["gt"] = gt_all
    df.to_csv("./cluster_results_seurat/seurat.csv")
    print("%s: ari:%.2f" % (root, ari))


def get_fovs(root="scanpy", num_fovs=30, key="scanpy", sep=","):
    pred_all = []
    gt_all = []
    index = []

    for i in range(1, num_fovs + 1):
        # Read csv for each fov
        csv = pd.read_csv(os.path.join(root, "fov%d.csv" % (i)), header=0, index_col=0, sep=sep)
        index.extend(list(csv.index))
        ind = csv["merge_cell_type"].isna()
        csv = csv[~ind]
        pred = csv[key].astype("category").cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred

        gt = csv["merge_cell_type"].astype("category")
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)

        id2cell = {}
        for g, gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g

        match = _hungarian_match(pred, gt_cat, len(set(pred)), len(set(gt_cat)))

        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc

        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)

    # Compute ARI and save results to csv
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df["pred"] = pred_all
    df["gt"] = gt_all
    df.to_csv("./cluster_results_stagate/stagate.csv")
    print("%s: ari:%.2f" % (root, ari))


# Compute ARI for different methods and lung cancer slides
# get_fovs(root="./cluster_results_scanpy/lung5-rep1", num_fovs=30)
# get_fovs(root="./cluster_results_scanpy/lung5-rep2", num_fovs=30)
# get_fovs(root="./cluster_results_scanpy/lung5-rep3", num_fovs=30)
# get_fovs(root="./cluster_results_scanpy/lung6", num_fovs=30)
# get_fovs(root="./cluster_results_scanpy/lung9-rep1", num_fovs=20)
# get_fovs(root="./cluster_results_scanpy/lung9-rep2", num_fovs=45)
# get_fovs(root="./cluster_results_scanpy/lung12", num_fovs=28)
get_fovs(root="./cluster_results_scanpy/lung13", num_fovs=20)


# get_fovs(root="./cluster_results_stagate/lung5-1", num_fovs=30, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung5-2", num_fovs=30, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung5-3", num_fovs=30, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung6", num_fovs=30, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung9-1", num_fovs=20, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung9-2", num_fovs=45, key="mclust")
# get_fovs(root="./cluster_results_stagate/lung12", num_fovs=28, key="mclust")
get_fovs(root="./cluster_results_stagate/lung13", num_fovs=20, key="mclust")

# get_fovs(root="./cluster_results_spagcn/lung5-rep1", num_fovs=30, key="refined_pred")
# get_fovs(root="./cluster_results_spagcn/lung5-rep2", num_fovs=30, key="refined_pred")
# get_fovs(root="./cluster_results_spagcn/lung5-rep3", num_fovs=30, key="refined_pred")
# get_fovs(root="./cluster_results_stlearn/lung6", num_fovs=30, key="refined_pred")
# get_fovs(root="./cluster_results_spagcn/lung9-rep1", num_fovs=20, key="refined_pred")
# get_fovs(root="./cluster_results_stlearn/lung9-2", num_fovs=45, key="refined_pred")
# get_fovs(root="./cluster_results_spagcn/lung12", num_fovs=28, key="refined_pred")
get_fovs(root="./cluster_results_spagcn/lung13", num_fovs=20, key="refined_pred")

# get_seurat(root="./cluster_results_seurat/lung5-1", num_fovs=30, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung5-1")
# get_seurat(root="./cluster_results_seurat/lung5-2", num_fovs=30, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung5-2")
# get_seurat(root="./cluster_results_seurat/lung5-3", num_fovs=30, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung5-3")
# get_seurat(root="./cluster_results_seurat/lung6", num_fovs=30, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung6")
# get_seurat(root="./cluster_results_seurat/lung9-1", num_fovs=20, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung9-1")
# get_seurat(root="./cluster_results_seurat/lung9-2", num_fovs=45, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung9-2")
# get_seurat(root="./cluster_results_seurat/lung12", num_fovs=28, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung12")
get_seurat(root="./cluster_results_seurat/lung13", num_fovs=20, key="seurat_clusters", sep="\t", dataroot="./cluster_results_stagate/lung13")

# get_stlearn(root="./cluster_results_stlearn/lung5-1", num_fovs=30)
# get_stlearn(root="./cluster_results_stlearn/lung5-2", num_fovs=30)
# get_stlearn(root="./cluster_results_stlearn/lung5-3", num_fovs=30)
# get_stlearn(root="./cluster_results_stlearn/lung6", num_fovs=30)
# get_stlearn(root="./cluster_results_stlearn/lung9-1", num_fovs=20)
# get_stlearn(root="./cluster_results_stlearn/lung9-2", num_fovs=45)
# get_stlearn(root="./cluster_results_stlearn/lung12", num_fovs=28)
get_stlearn(root="./cluster_results_stlearn/lung13", num_fovs=20)

# get_bayesspace(root="./cluster_results_bayesspace/lung5-1/NA", num_fovs=30)
# get_bayesspace(root="./cluster_results_bayesspace/lung5-2/NA", num_fovs=30)
# get_bayesspace(root="./cluster_results_bayesspace/lung5-3/NA", num_fovs=30)
# get_bayesspace(root="./cluster_results_bayesspace/lung6/NA", num_fovs=30)
# get_bayesspace(root="./cluster_results_bayesspace/lung9-1/NA", num_fovs=20)
# get_bayesspace(root="./cluster_results_bayesspace/lung9-2/NA", num_fovs=45)
# get_bayesspace(root="./cluster_results_bayesspace/lung12/NA", num_fovs=28)
get_bayesspace(root="./cluster_results_bayesspace/lung13/NA", num_fovs=20)
