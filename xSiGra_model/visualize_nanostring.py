import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from train_transformer import test_nano_fov
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--root", type=str, default="/N/project/st_brain/abudhkar/dataset/nanostring/lung13")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--id", type=str, default="fov1")
parser.add_argument("--img_name", type=str, default="F001")
parser.add_argument("--neurons", type=str, default="512,30")
parser.add_argument("--num_layers", type=str, default="2")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_path", type=str, default="../checkpoint/nanostring_train_lung13/")
parser.add_argument("--ncluster", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--use_gray", type=float, default=0)
parser.add_argument("--test_only", type=int, default=1)
parser.add_argument("--pretrain", type=str, default="final_100_0.pth")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--img_size", type=str, default="60,60")
parser.add_argument("--cluster_method", type=str, default="leiden")
parser.add_argument("--num_fov", type=int, default=20)
parser.add_argument("--dataset", type=str, default="lung13")

opt = parser.parse_args()

# Set seed
root = opt.root
seed = opt.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Function to process data
def gen_adatas(root, id, img_name):
    adata = sc.read(os.path.join(root, id, "sampledata.h5ad"))
    adata.var_names_make_unique()
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


ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
img_names = ["F00" + str(i) for i in range(1, 10)]
img_names = img_names + ["F0" + str(i) for i in range(10, int(opt.num_fov) + 1)]

adatas = list()
for id, name in zip(ids, img_names):
    adata = gen_adatas(opt.root, id, name)
    adatas.append(adata)

sp = os.path.join(opt.save_path, "all")


## Uncomment to select model from scratch
# Choose best model
# adata_pred = test_nano_fov(
#     opt,
#     adatas,
#     hidden_dims=opt.neurons,
#     random_seed=opt.seed,
#     save_path=sp,
# )

# sc.pp.neighbors(adata_pred, opt.ncluster, use_rep="pred")
# def res_search(adata_pred, ncluster, seed, iter=200):
#     start = 0
#     end = 3
#     i = 0
#     while start < end:
#         if i >= iter:
#             return res
#         i += 1
#         res = (start + end) / 2

#         random.seed(seed)
#         os.environ["PYTHONHASHSEED"] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
#         count = len(set(adata_pred.obs["leiden"]))
#         if count == ncluster:
#             return res
#         if count > ncluster:
#             end = res
#         else:
#             start = res
#     raise NotImplementedError()

# # Perform leiden clustering
# res = res_search(adata_pred, opt.ncluster, opt.seed)

## Comment below lines if selecting model above
# Use our provided saved results for Lung 13  with best resolution or use your computed results and resolution (change)
adata_pred = sc.read("../saved_adata/" + opt.dataset + "_adata_pred.h5ad")
res = 0.3681250000000001

# Perform leiden clustering
sc.tl.leiden(adata_pred, resolution=res, key_added="leiden", random_state=opt.seed)
obs_df = adata_pred.obs.dropna()

ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)

ARI2 = adjusted_rand_score(obs_df["merge_cell_type"], obs_df["leiden"])
print("ARI2: %.2f" % ARI2)

labels = obs_df["merge_cell_type"]


# Match clusters to cell types
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

match = _hungarian_match(adata_pred.obs["leiden"].astype(np.int8), gt.astype(np.int8), opt.ncluster, 8)

# Leiden to ground truth cell type matching
dict_mapping = {}

# Leiden cluster cell type
dict_name = {}

# Ground truth cluster cell type
dict_gtname = {}
for i in gt:
    dict_gtname[i] = le.classes_[i]

dict_gtname = {}
for i in match:
    dict_mapping[str(i[0])] = i[1]
    dict_name[i[0]] = le.classes_[i[1]]
    dict_gtname[i[1]] = le.classes_[i[0]]

obs_df = obs_df.replace({"leiden": dict_mapping})
adata_pred.obs["leiden"] = obs_df["leiden"]
for val in set(adata_pred.obs["leiden"]):
    dict_name[val] = le.classes_[val]

ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)

leiden = obs_df["leiden"].tolist()
labels = obs_df["merge_cell_type"].tolist()

le = preprocessing.LabelEncoder()
gt = le.fit_transform(labels)


match = _hungarian_match(adata_pred.obs["leiden"].astype(np.int8), gt.astype(np.int8), opt.ncluster, 8)

# Leiden to ground truth cell type matching
dict_mapping = {}

# Leiden cluster cell type
dict_name = {}

# Ground truth cluster cell type
dict_gtname = {}
for i in gt:
    dict_gtname[i] = le.classes_[i]

dict_gtname = {}
for i in match:
    dict_mapping[str(i[0])] = i[1]
    dict_name[i[0]] = le.classes_[i[1]]
    dict_gtname[i[1]] = le.classes_[i[0]]

obs_df = obs_df.replace({"leiden": dict_mapping})
adata_pred.obs["leiden"] = obs_df["leiden"]
for val in set(adata_pred.obs["leiden"]):
    dict_name[val] = le.classes_[val]

ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)

leiden = obs_df["leiden"].tolist()
labels = obs_df["merge_cell_type"].tolist()

le = preprocessing.LabelEncoder()
gt = le.fit_transform(labels)

# Plot ground truth and computed clusters
cell_type = list(set(adata_pred.obs["merge_cell_type"]))
ground_truth = [i for i in range(len(cell_type))]
gt = np.zeros(adata_pred.obs["merge_cell_type"].shape)
for i in range(len(ground_truth)):
    ct = cell_type[i]
    idx = adata_pred.obs["merge_cell_type"] == ct
    gt[idx] = i
gt = gt.astype(np.int32)

pred = adata_pred.obs["leiden"].to_numpy().astype(np.int32)

# Match clusters to cell types
match = _hungarian_match(pred, gt, len(set(pred)), len(set(gt)))

# Color code for cell types
colors = {
    "lymphocyte": "#fa9898ff",
    "Mcell": "#6879ccff",
    "myeloid": "#6879ccff",
    "tumors": "#bce1f5ff",
    "epithelial": "#c2d64fff",
    "mast": "#fccc3dff",
    "endothelial": "#2dc2d6ff",
    "fibroblast": "#42c2b5ff",
    "neutrophil": "#c174cfff",
}
cs = ["" for i in range(pred.shape[0])]
gt_cs = ["" for i in range(pred.shape[0])]

for ind, j in enumerate(adata_pred.obs["merge_cell_type"].tolist()):
    gt_cs[ind] = colors[j]

for outc, gtc in match:
    idx = pred == outc
    print(idx)
    for j in range(len(idx)):
        if idx[j]:
            cs[j] = colors[cell_type[gtc]]

colors = {
    0: "#2dc2d6ff",
    1: "#c2d64fff",
    1: "#c2d64fff",
    2: "#42c2b5ff",
    3: "#fa9898ff",
    4: "#fccc3dff",
    5: "#6879ccff",
    6: "#c174cfff",
    7: "#bce1f5ff",
}
for ind, j in enumerate(adata_pred.obs["leiden"].tolist()):
    cs[ind] = colors[int(j)]
adata_pred.obs["cmap"] = cs
adata_pred.obs["gtcmap"] = gt_cs

# Plot spatial figures
genedf = sc.get.obs_df(adata_pred, keys=["leiden", "cmap", "gtcmap"])
cxg, cyg = adata_pred.obs["cx_g"], adata_pred.obs["cy_g"]
colors = adata_pred.obs["gtcmap"]
fig, axs = plt.subplots(4, 1, figsize=(10, 40))
# axs[0].invert_yaxis()
axs[0].scatter(cxg, cyg, c=colors, s=0.25)
axs[0].axis("off")
axs[0].set_title("ground truth")
extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
if not os.path.exists(os.path.join("../cluster_results_gradcam_gt1/" + opt.dataset)):
    os.makedirs(os.path.join("../cluster_results_gradcam_gt1/" + opt.dataset))

fig.savefig("../cluster_results_gradcam_gt1/" + opt.dataset + "/gt.png", bbox_inches=extent)

colors = adata_pred.obs["cmap"]
# axs[1].invert_yaxis()
axs[1].scatter(cxg, cyg, c=colors, s=0.25)
axs[1].axis("off")
axs[1].set_title("SiGra")
extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("../cluster_results_gradcam_gt1/" + opt.dataset + "/pred.png", bbox_inches=extent)

# Violin plot for marker genes in raw and enchanced data

markers = ["CD68", "CD163", "MGP", "SERPINA1", "KRT7", "DCN", "CD2", "SRGN", "EPCAM"]
adata_copy = adata_pred.copy()
order = None

if "Mcell" in adata_pred.obs["merge_cell_type"].tolist():
    order = [
        "Mcell",
        "endothelial",
        "epithelial",
        "fibroblast",
        "lymphocyte",
        "mast",
        "neutrophil",
        "tumors",
    ]
else:
    order = [
        "myeloid",
        "endothelial",
        "epithelial",
        "fibroblast",
        "lymphocyte",
        "mast",
        "neutrophil",
        "tumors",
    ]
adata_copy.obs["merge_cell_type"] = adata_copy.obs["merge_cell_type"].astype("category")
adata_copy.obs["merge_cell_type"].cat.reorder_categories(order, inplace=True)

# Violin plot
sc.pl.stacked_violin(
    adata_copy,
    markers,
    groupby="merge_cell_type",
    dendrogram=False,
    layer="recon",
    save=opt.dataset + "_enhanced.pdf",
    title="Enhanced",
    swap_axes=True,
    order=order,
)
sc.pl.stacked_violin(
    adata_copy,
    markers,
    groupby="merge_cell_type",
    dendrogram=False,
    save=opt.dataset + "_raw.pdf",
    title="Raw",
    swap_axes=True,
    order=order,
)

count = 0
start = 0
aris = []
ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
img_names = ["F00" + str(i) for i in range(1, 10)]
img_names = img_names + ["F00" + str(i) for i in range(10, opt.num_fov + 1)]

colors = {
    "lymphocyte": "#fa9898ff",
    "Mcell": "#6879ccff",
    "myeloid": "#6879ccff",
    "tumors": "#bce1f5ff",
    "epithelial": "#c2d64fff",
    "mast": "#fccc3dff",
    "endothelial": "#2dc2d6ff",
    "fibroblast": "#42c2b5ff",
    "neutrophil": "#c174cfff",
}

adata_new = adata_pred[~adata_pred.obs["merge_cell_type"].isin(["tumors"]), :]

# Plot UMAP
sc.pp.neighbors(adata_new, use_rep="pred")
sc.tl.umap(adata_new)
plt.rcParams["figure.figsize"] = (3, 3)
if not os.path.exists("./figures/"):
    os.makedirs("./figures/")
save_path = "./figures/"
sc.settings.figdir = save_path
ax = sc.pl.umap(adata_new, color=["merge_cell_type"], show=False, title="Enhanced", palette=colors)
plt.savefig(os.path.join(save_path, opt.dataset + "_umap_gt.pdf"), bbox_inches="tight")

sc.pp.neighbors(adata_new, use_rep="gene_pred")
sc.tl.umap(adata_new)
plt.rcParams["figure.figsize"] = (3, 3)
save_path = "./figures/"
sc.settings.figdir = save_path
ax = sc.pl.umap(adata_new, color=["merge_cell_type"], show=False, title="Raw", palette=colors)
plt.savefig(os.path.join(save_path, opt.dataset + "_umap_gt_enh.pdf"), bbox_inches="tight")
