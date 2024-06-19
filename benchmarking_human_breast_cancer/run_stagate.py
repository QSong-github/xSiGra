import argparse
import anndata
import os

import numpy as np
import pandas as pd
import random
import scanpy as sc
import STAGATE_pyG as STAGATE
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2
from tqdm import tqdm

cudnn.deterministic = True
cudnn.benchmark = True

# Set R paths and seed
os.environ["R_HOME"] = "/N/soft/rhel8/r/gnu/4.2.1_X11/lib64/R/"
os.environ["R_USER"] = "/N/soft/rhel8/r/gnu/4.2.1_X11/lib64/R/"
os.environ["LD_LIBRARY_PATH"] = "/N/soft/rhel8/r/gnu/4.2.1_X11/lib64/R/lib/"
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def gen_adatas(adata):
    print(id)

    # Read data
    adata = adata
    adata.var_names_make_unique()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    arr = adata.obsm["spatial"]

    # Create a DataFrame from arr
    df1 = pd.DataFrame(arr, index=adata.obs.index, columns=["cx", "cy"])
    
    # Calculate spatial network
    df = pd.DataFrame(index=adata.obs.index)
    df["cx"] = df1["cx"]
    df["cy"] = df1["cy"]
    arr = df.to_numpy()
    adata.obsm["spatial"] = arr

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=400)
    return adata


def train(
    adata,
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
    parser.add_argument("--path", type=str, default="../dataset/breast_invasive_carcinoma/")
    parser.add_argument("--save_path", type=str, default="./HBC_STAGATE/")
    parser.add_argument("--ncluster", type=int, default=8)

    opt = parser.parse_args(args=[])
    n_epochs = 1000
    path = opt.path

    
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Read data
    adata = sc.read_visium(path, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    adata.X = adata.X.A
        
    ncluster = opt.ncluster

    adata = gen_adatas(adata)
    adata = train(adata, save_path=save_path, n_epochs=n_epochs)
    
    # Clustering
    adata = STAGATE.mclust_R(adata, num_cluster=ncluster, used_obsm="STAGATE",random_seed=1234)   
    obs_df = adata.obs.dropna()

    from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score

    silhouette = silhouette_score(adata.obsm['STAGATE'], obs_df["mclust"])
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(adata.obsm['STAGATE'], obs_df["mclust"])
    print('Davies-Bouldin Score: %.2f' % davies_bouldin)

    calinski = calinski_harabasz_score(adata.obsm["STAGATE"], obs_df['mclust'])
    print('Calinski Score: %.2f' % calinski)

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax=sc.pl.spatial(adata, color=['mclust'], title=['STAGATE'], show=False)
    plt.savefig(os.path.join(save_path, 'STAGATE_spatial.pdf'), bbox_inches='tight')
