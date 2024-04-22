import argparse
import os
import random

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scanpy as sc
import STAGATE_pyG as STAGATE
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

cudnn.deterministic = True
cudnn.benchmark = True

# Set R path and seed
os.environ["R_HOME"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["R_USER"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib/"
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def gen_adatas(root, id):
    print(id)

    # Read data
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))
    adata.var_names_make_unique()

    # Read annotation
    Ann_df = pd.read_csv("%s/%s/annotation.txt" % (opt.root, opt.id), sep="\t", header=None, index_col=0)
    Ann_df.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]

    ind = adata.obs["Ground Truth"].isna()
    adata = adata[~ind]

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Calculate spatial network
    coors = adata.obsm["spatial"]
    adata.obs["px"] = coors[:, 0]
    adata.obs["py"] = coors[:, 1]
    df = pd.DataFrame(index=adata.obs.index)
    df["cx"] = adata.obs["px"]
    df["cy"] = adata.obs["py"]
    arr = df.to_numpy()
    adata.obsm["spatial"] = arr

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=80)
    return adata


def train(
    adata,
    id,
    hidden_dims=[512, 30],
    n_epochs=1000,
    lr=0.001,
    key_added="STAGATE",
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    weight_decay=0.0001,
    save_path="./STAGATE/lung9-2/",
):
    seed = random_seed

    # Set seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = STAGATE.utils.Transfer_pytorch_Data(adata)

    # Get model instance
    model = STAGATE.STAGATE(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train model
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "STAGATE_%s.pth" % (id)))

    # Get predictions
    with torch.no_grad():
        model.eval()
        pred = None
        pred_out = None
        z, out = model(data.x, data.edge_index)
        pred = z
        pred = pred.cpu().detach().numpy()
        pred_out = out.cpu().detach().numpy().astype(np.float32)
        pred_out[pred_out < 0] = 0

        adata.obsm[key_added] = pred
        adata.obsm["recon"] = pred_out

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/nanostring")
    parser.add_argument("--id", type=str, default="151676")
    parser.add_argument("--save_path", type=str, default="./stagate/")
    parser.add_argument("--ncluster", type=int, default=7)
    opt = parser.parse_args()

    root = opt.root
    save_path = opt.save_path
    n_epochs = 1000
    ids = [opt.id]
    adatas = list()
    for id in ids:
        adata = gen_adatas(root, id)

        # Train model
        adata = train(adata, id, save_path=save_path, n_epochs=n_epochs)

        # Clustering
        adata = STAGATE.mclust_R(adata, num_cluster=opt.ncluster, used_obsm="STAGATE")

        obs_df = adata.obs.dropna()

        # Compute ARI
        ARI = adjusted_rand_score(obs_df["mclust"], obs_df["Ground Truth"])
        print(ARI)
        adatas.append(adata)

        labels = obs_df["Ground Truth"]

    adata1 = anndata.concat(adatas)
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))
    Ann_df = pd.read_csv("%s/%s/annotation.txt" % (root, id), sep="\t", header=None, index_col=0)
    Ann_df.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]

    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.pca(adata, svd_solver="arpack")
    adata.obs["Ground Truth"] = adata.obs["Ground Truth"]

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

    adata.obs["leiden"] = adata1.obs["mclust"]
    dict1 = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
    }

    obs_df = adata.obs.dropna()

    df = obs_df["leiden"].tolist()
    df = [dict1[x] for x in df]

    obs_df["leiden"] = df

    ARI = adjusted_rand_score(obs_df["leiden"], labels)
    print(ARI)

    labels = obs_df["Ground Truth"]

    # Match cluster to ground truth
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

    match = _hungarian_match(obs_df["leiden"].astype(np.int8), gt.astype(np.int8), opt.ncluster, opt.ncluster)

    dict1 = {}
    dict_name = {}
    for i in match:
        dict1[str(i[0])] = i[1]
        dict1[int(i[0])] = i[1]
        dict_name[i[1]] = le.classes_[i[0]]

    # Replace matched leiden with ground truth
    obs_df = obs_df.replace({"leiden": dict1})

    adata.obs["leiden"] = obs_df["leiden"]

    plt.rcParams["figure.figsize"] = (3, 3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path

    # Save spatial figures
    ax = sc.pl.spatial(adata, color=["Ground Truth"], title=["Ground Truth"], show=False)
    plt.savefig(os.path.join(save_path, "gt_spatial1.pdf"), bbox_inches="tight")

    df = adata.obs["leiden"]

    df = df.fillna(7)
    leiden = df.tolist()
    leiden = [int(x) for x in leiden]
    leiden = [str(x) for x in leiden]
    adata.obs["leiden"] = leiden
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.spatial(adata, color=["leiden"], title=["Stagate (ARI=%.2f)" % (ARI)], show=False)
    plt.savefig(os.path.join(save_path, "spatial1.pdf"), bbox_inches="tight")

    # Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title="Stagate", legend_loc="on data")
    plt.savefig(os.path.join(save_path, "umap_final1.pdf"), bbox_inches="tight")

    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["Ground Truth"])
    print("ARI: %.2f" % ARI)

    obs_df = adata.obs
    df = pd.DataFrame(index=adata.obs.index)
    df["mclust"] = adata1.obs["mclust"]

    # Save results to csv
    df["Ground Truth"] = adata.obs["Ground Truth"]
    df.to_csv(os.path.join(save_path, "%s.csv" % id))

    df = pd.DataFrame(adata.obs["leiden"], index=adata.obs.index)
    df.to_csv(os.path.join(save_path, "%s_STAGATE.csv" % id))
