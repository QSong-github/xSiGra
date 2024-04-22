import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from train_transformer import test_img, train_img
from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, seed_everything

# Set R path and seed
os.environ["R_HOME"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["R_USER"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib"
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@torch.no_grad()
def infer(opt, result_path="../10x_results/"):
    seed_everything(opt.seed)

    # Read data
    adata = sc.read(os.path.join(opt.root, opt.id, "sampledata.h5ad"))

    # Pre-process
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get ground truth
    Ann_df = pd.read_csv("%s/%s/annotation.txt" % (opt.root, opt.id), sep="\t", header=None, index_col=0)
    Ann_df.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]

    sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")

    img = cv2.imread(os.path.join(opt.root, opt.id, "spatial/full_image.tif"))
    if opt.use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    for coor in adata.obsm["spatial"]:
        py, px = coor
        img_p = img[:, px - 25 : px + 25, py - 25 : py + 25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm["imgs"] = df

    Cal_Spatial_Net(adata, rad_cutoff=150)

    # Set proper path for R
    os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib/"
    os.environ["PYTHONHASHSEED"] = "1234"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["R_HOME"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"

    model_path = os.path.join(os.path.join(opt.save_path, opt.id))

    # Test model
    adata = test_img(adata, save_path=model_path, hidden_dims=[512, 30])

    # Perform clustering
    adata = mclust_R(adata, used_obsm="pred", num_cluster=opt.ncluster)
    obs_df = adata.obs.dropna()

    # Compute ARI
    ARI = adjusted_rand_score(obs_df["mclust"], obs_df["Ground Truth"])
    print("ari is %.2f" % (ARI))

    labels = obs_df["Ground Truth"]
    obs_df["leiden"] = obs_df["mclust"]

    map_gt = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
    }

    df = obs_df["leiden"].tolist()
    df = [int(x) for x in df]
    df = [map_gt[x] for x in df]

    obs_df["leiden"] = df

    # Match cluster to ground truth
    def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
        num_samples = flat_target.shape[0]
        num_k = preds_k
        num_k_target = target_k
        num_correct = np.zeros((num_k, num_k_target))
        for c1 in range(num_k):
            for c2 in range(num_k_target):
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

    match = _hungarian_match(obs_df["leiden"].astype(np.int8), gt.astype(np.int8), opt.ncluster, opt.ncluster)

    map_cluster = {}
    dict_name = {}
    for i in match:
        map_cluster[int(i[0])] = int(i[1])
        map_cluster[str(i[0])] = int(i[1])
        dict_name[i[1]] = le.classes_[i[0]]

    obs_df = obs_df.replace({"leiden": map_cluster})

    adata.obs["leiden"] = obs_df["leiden"].astype(np.int8)

    # Plot spatial figure
    plt.rcParams["figure.figsize"] = (3, 3)
    save_path = os.path.join(result_path, opt.id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Slice 151676"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial.pdf"), bbox_inches="tight")

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["xSiGra (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial.pdf"), bbox_inches="tight")

    # Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="pred")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(
        adata,
        color=["Ground Truth"],
        show=False,
        title="Ground truth",
        legend_loc="on data",
    )
    plt.savefig(os.path.join(save_path, "umap_final1.pdf"), bbox_inches="tight")

    sc.pp.neighbors(adata, n_neighbors=20, use_rep="pred")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="Our method", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final2.pdf"), bbox_inches="tight")

    return ARI


def train(opt, r):
    # Set seed
    seed_everything(opt.seed)

    # Read data
    adata = sc.read(os.path.join(opt.root, opt.id, "sampledata.h5ad"))

    # Pre-process
    adata.var_names_make_unique()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Read ground truth
    Ann_df = pd.read_csv("%s/%s/annotation.txt" % (opt.root, opt.id), sep="\t", header=None, index_col=0)
    Ann_df.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]

    img = cv2.imread(os.path.join(opt.root, opt.id, "spatial/full_image.tif"))
    if opt.use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    for coor in adata.obsm["spatial"]:
        py, px = coor
        img_p = img[:, px - 25 : px + 25, py - 25 : py + 25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm["imgs"] = df

    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    sp = os.path.join(opt.save_path, opt.id)
    if not os.path.exists(sp):
        os.makedirs(sp)

    # Set proper path for R
    os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib/"
    os.environ["PYTHONHASHSEED"] = "1234"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Train model
    adata = train_img(
        adata,
        hidden_dims=[512, 30],
        n_epochs=opt.epochs,
        lr=opt.lr,
        random_seed=opt.seed,
        save_path=sp,
        repeat=r,
    )

    os.environ["R_HOME"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"

    # Clustering
    adata = mclust_R(adata, used_obsm="pred", num_cluster=opt.ncluster)
    obs_df = adata.obs.dropna()

    # Compute ARI
    ARI = adjusted_rand_score(obs_df["mclust"], obs_df["Ground Truth"])
    print("ari is %.2f" % (ARI))
    return ARI


def train_10x(opt):
    # Set correct path
    os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib/"

    # Set cluster method mclust
    opt.cluster_method = "mclust"
    if opt.test_only:
        ARI = infer(opt)
        print("ARI is %.2f" % (ARI))
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        ARI = train(opt, 0)

    return 0
