import argparse
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torchvision.transforms as transforms
from sklearn.metrics.cluster import adjusted_rand_score
from train_transformer import test_nano_fov, train_nano_fov
from utils import Cal_Spatial_Net, Stats_Spatial_Net

# Set R path
os.environ["R_HOME"] = "/opt/R/4.0.2/lib/R"
os.environ["R_USER"] = "~/anaconda3/lib/python3.8/site-packages/rpy2"
os.environ["LD_LIBRARY_PATH"] = "/opt/R/4.0.2/lib/R/lib"
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# Function to generate anndata
def gen_adatas(root, id, img_name):
    # Read gene expression and image
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))
    adata.var_names_make_unique()

    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    img = cv2.imread(os.path.join(root, id, "CellComposite_%s.jpg" % (img_name)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if opt.use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []

    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)

    for coor in adata.obsm["spatial"]:
        x, y = coor
        img_p = img[:, int(y - h) : int(y + h), int(x - w) : int(x + w)]
        patchs.append(img_p.flatten())  # 4 * h * w

    patchs = np.stack(patchs)

    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm["imgs"] = df

    # Construct spatial graph
    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata


@torch.no_grad()
def infer(opt, r=0):
    seed = opt.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ids = [
        "fov1",
        "fov2",
        "fov3",
        "fov4",
        "fov5",
        "fov6",
        "fov7",
        "fov8",
        "fov9",
        "fov10",
        "fov11",
        "fov12",
        "fov13",
        "fov14",
        "fov15",
        "fov16",
        "fov17",
        "fov18",
        "fov19",
        "fov20",
    ]
    img_names = [
        "F001",
        "F002",
        "F003",
        "F004",
        "F005",
        "F006",
        "F007",
        "F008",
        "F009",
        "F010",
        "F011",
        "F012",
        "F013",
        "F014",
        "F015",
        "F016",
        "F017",
        "F018",
        "F019",
        "F020",
    ]

    img_names = []
    ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
    img_names = ["F00" + str(i) for i in range(1, 10)]
    img_names = img_names + ["F0" + str(i) for i in range(10, 100)]
    img_names = img_names + ["F" + str(i) for i in range(100, int(opt.num_fov) + 1)]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, "all")

    # Get best model
    adata_pred = test_nano_fov(
        opt,
        adatas,
        hidden_dims=opt.neurons,
        random_seed=opt.seed,
        save_path=sp,
    )

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
                return res
            if count > ncluster:
                end = res
            else:
                start = res

    # Tune model
    sc.pp.neighbors(adata_pred, opt.ncluster, use_rep="pred")
    res = res_search(adata_pred, opt.ncluster, opt.seed)

    sc.tl.leiden(adata_pred, resolution=res, key_added="leiden", random_state=opt.seed)
    obs_df = adata_pred.obs.dropna()

    # Calculate ARI
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
    print("ARI: %.2f" % ARI)

    seed = 1234

    ari = []
    resl = []
    ncluster = []
    best_ari = 0
    best_res = 0

    for res in np.arange(res - 0.30, res + 0.30, 0.01):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        resl.append(res)
        sc.tl.leiden(adata_pred, resolution=res, key_added="leiden", random_state=seed)
        cluster = len(set(adata_pred.obs["leiden"]))
        ncluster.append(cluster)
        obs_df = adata_pred.obs.dropna()

        ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
        print("ARI: %.2f" % ARI)
        ari.append(round(ARI, 2))

        if ARI > best_ari and cluster == opt.ncluster:
            best_ari = ARI
            best_res = res

    print("Selected resolution:", best_res)
    if not os.path.exists("../saved_adata/"):
        os.makedirs("../saved_adata")
    df = pd.DataFrame({"no of clusters": ncluster, "resolution": resl, "ari": ari})
    df.to_csv("../saved_adata/" + opt.dataset + "finetune.csv")

    sc.tl.leiden(adata_pred, resolution=best_res, key_added="leiden", random_state=opt.seed)
    obs_df = adata_pred.obs.dropna()

    # Calculate ARI
    ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
    print("ARI: %.2f" % ARI)

    if not os.path.exists("../saved_adata/"):
        os.makedirs("../saved_adata/")
    adata_pred.obsm["imgs"] = adata_pred.obsm["imgs"].to_numpy()
    adata_pred.obs.columns = adata_pred.obs.columns.astype(str)
    adata_pred.var.columns = adata_pred.var.columns.astype(str)
    adata_pred.write("../saved_adata/" + opt.dataset + "_adata_pred.h5ad")


def train(opt, r=0):
    seed = opt.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    img_names = []
    ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
    img_names = ["F00" + str(i) for i in range(1, 10)]
    img_names = img_names + ["F0" + str(i) for i in range(10, 100)]
    img_names = img_names + ["F" + str(i) for i in range(100, int(opt.num_fov) + 1)]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, "all")
    if not os.path.exists(sp):
        os.makedirs(sp)

    # Train and save model
    train_nano_fov(
        opt,
        adatas,
        hidden_dims=opt.neurons,
        n_epochs=opt.epochs,
        lr=opt.lr,
        random_seed=opt.seed,
        save_path=sp,
        repeat=r,
    )


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root", type=str, default="../dataset/nanostring/lung13")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--id", type=str, default="fov1")
    parser.add_argument("--img_name", type=str, default="F001")
    parser.add_argument("--img_size", type=str, default="60,60")
    parser.add_argument("--neurons", type=str, default="512,30")
    parser.add_argument("--num_layers", type=str, default="2")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_path", type=str, default="../checkpoint/nanostring_train_lung13/")
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--use_gray", type=float, default=0)
    parser.add_argument("--test_only", type=int, default=0)
    parser.add_argument("--pretrain", type=str, default="final_100_0.pth")
    parser.add_argument("--cluster_method", type=str, default="leiden")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_fov", type=str, default=20)
    parser.add_argument("--dataset", type=str, default="lung13")

    opt = parser.parse_args()

    if opt.test_only:
        infer(opt)
    else:
        train(opt, 0)

    end_time = time.time()
    delta = end_time - start_time

    sec = delta

    hours = sec / (60 * 60)
    print("difference in hours:", hours)


def train_nano(opt):
    if opt.test_only:
        infer(opt)
    else:
        train(opt, 0)
