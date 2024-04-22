import argparse
import os

import numpy as np
import pandas as pd
import random
import scanpy as sc
import STAGATE_pyG as STAGATE
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

cudnn.deterministic = True
cudnn.benchmark = True

# Set R paths and seed
os.environ["R_HOME"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["R_USER"] = "/N/soft/rhel7/r/4.2.1/lib64/R/"
os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel7/r/4.2.1/lib64/R/lib"
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def gen_adatas(root, id):
    print(id)

    # Read data
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))
    adata.var_names_make_unique()
    ind = adata.obs["merge_cell_type"].isna()
    adata = adata[~ind]

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Calculate spatial network
    df = pd.DataFrame(index=adata.obs.index)
    df["cx"] = adata.obs["cx"]
    df["cy"] = adata.obs["cy"]
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

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = STAGATE.utils.Transfer_pytorch_Data(adata)

    # Create model instance
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

    # Get cluster predictions
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
    parser.add_argument("--save_path", type=str, default="./stagate/")
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--num_fov", type=int, default=20)
    opt = parser.parse_args()

    root = opt.root
    save_path = opt.save_path
    n_epochs = 1000
    num_fov = opt.num_fov
    ids = ["fov%d" % i for i in range(1, num_fov + 1)]
    adatas = list()

    # For each fov
    for id in ids:
        adata = gen_adatas(root, id)

        # Train model
        adata = train(adata, id, save_path=save_path, n_epochs=n_epochs)

        # Clustering
        adata = STAGATE.mclust_R(adata, num_cluster=opt.ncluster, used_obsm="STAGATE")
        obs_df = adata.obs.dropna()

        # Compute ARI
        ARI = adjusted_rand_score(obs_df["mclust"], obs_df["merge_cell_type"])
        print(ARI)

        df = pd.DataFrame(index=adata.obs.index)
        df["mclust"] = adata.obs["mclust"]
        df["merge_cell_type"] = adata.obs["merge_cell_type"]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save clusters to csv
        df.to_csv(os.path.join(save_path, "%s.csv" % id))
        df = pd.DataFrame(adata.obsm["STAGATE"], index=adata.obs.index)
        df.to_csv(os.path.join(save_path, "%s_STAGATE.csv" % id))
