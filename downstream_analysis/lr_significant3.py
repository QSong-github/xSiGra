import argparse
import os
import sys
import random

import anndata
import cv2
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torchvision.transforms as transforms
from scipy import stats
from statsmodels.stats import multitest

sys.path.append("../")
from xSiGra_model.utils import Cal_Spatial_Net, Stats_Spatial_Net

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="../dataset/nanostring/lung13")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--use_gray", type=float, default=0)
parser.add_argument("--img_size", type=str, default="60,60")
parser.add_argument("--num_fov", type=int, default=20)

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


# Function to process files
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

    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata


ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
img_names = ["F00" + str(i) for i in range(1, 10)]
img_names = img_names + ["F0" + str(i) for i in range(10, int(opt.num_fov) + 1)]

# Use fov subset for parallel computation
ids = ["fov16", "fov17", "fov18", "fov19", "fov20"]
img_names = ["F016", "F017", "F018", "F019", "F020"]
adatas = list()
exp = pd.DataFrame()
for id, name in zip(ids, img_names):
    adata = gen_adatas(opt.root, id, name)
    if "_" not in adata.to_df().index[0]:
        # Format cell_id to make it in the form fov_cell_id "1_322"
        fov_exp = adata.to_df()
        fov_exp.index = [id[3:] + "_" + x for x in adata.to_df().index]
        exp = pd.concat([exp, fov_exp])
    adatas.append(adata)

# Create Spatial graph
graph = pd.DataFrame()
fov = 16
for adata in adatas:
    G_df = adata.uns["Spatial_Net"].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
    G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)
    G_df["fov"] = fov
    fov += 1
    graph = pd.concat([graph, G_df], ignore_index=True)

graph["Cell1"] = graph["fov"].astype("str") + "_" + graph["Cell1"].astype("str")
graph["Cell2"] = graph["fov"].astype("str") + "_" + graph["Cell2"].astype("str")

adata = anndata.concat(adatas)

# Raw expression
exp_columns = [x + " imp score" for x in list(exp.columns)]
exp_columns = exp_columns
exp.columns = exp_columns


# Read known ligand-receptor interactions
lr = pd.read_csv("LRdb_saved_files.csv")
ligands = lr["ligand"]
receptors = lr["receptor"]


def check(start, end, scores, pairs1, lr_score1, lr_score2, cci_results):
    # Read known LR pairs
    lr = pd.read_csv("LRdb_saved_files.csv")
    ligands = list(lr["ligand"])
    receptors = list(lr["receptor"])

    # Subseting to avoid out of memory
    ligands = ligands[start:end]
    receptors = receptors[start:end]

    # Read importance score
    celltypeimp1 = pd.read_csv("../cluster_results_gradcam_gt1/lung13/cluster7_tumors.csv")
    celltypeimp2 = pd.read_csv("../cluster_results_gradcam_gt1/lung13/cluster2_fibroblast.csv")

    if "_" not in str(celltypeimp1.iat[2, 1]):
        celltypeimp1["cellID"] = celltypeimp1["fov"].astype("str") + "_" + celltypeimp1["cellID"].astype("str")
        celltypeimp2["cellID"] = celltypeimp2["fov"].astype("str") + "_" + celltypeimp2["cellID"].astype("str")

    # Get gene names
    list1 = celltypeimp1.columns
    list1 = [x.split(" ")[0] for x in list1]

    list2 = celltypeimp2.columns
    list2 = [x.split(" ")[0] for x in list2]

    # LR pairs
    pairs = [i + "_" + j for i, j in zip(ligands, receptors) if i in list1 and j in list2]

    del list1
    del list2

    cell1 = graph["Cell1"]
    cell2 = graph["Cell2"]

    # For each cell neighbor pair, create space for each ligand receptor combination in the dataframe
    cell1 = cell1.loc[cell1.index.repeat(len(pairs))]
    cell2 = cell2.loc[cell2.index.repeat(len(pairs))]

    length = len(graph["Cell1"])
    final_df = pd.DataFrame()
    final_df["Cell1"] = cell1
    final_df["Cell2"] = cell2

    del cell1
    del cell2

    # Select genes that are present in dataset
    ligands = [x.split("_")[0] for x in pairs]
    receptors = [x.split("_")[1] for x in pairs]

    # Multiple by number of cells to create proper dataframe for further processing
    ligands = ligands * length
    ligands = [x + " imp score" for x in ligands]
    receptors = receptors * length
    receptors = [x + " imp score" for x in receptors]

    final_df["ligands"] = ligands
    final_df["receptor"] = receptors

    # Find cell type from ground truth data
    final_df["celltype1"] = final_df["Cell1"].map(celltypeimp1.set_index("cellID")["cell_type"])
    final_df = final_df.dropna()

    # Find cell type from ground truth data
    final_df["celltype2"] = final_df["Cell2"].map(celltypeimp2.set_index("cellID")["cell_type"])
    final_df = final_df.dropna()
    del celltypeimp1
    del celltypeimp2

    # Method 1: group by ligand -celltype receptor -celltype
    final_df["ligand-receptor"] = final_df["ligands"] + "_" + final_df["celltype1"] + "_" + final_df["receptor"] + "_" + final_df["celltype2"]

    # Method 2: group by ligand receptor
    # final_df["ligand-receptor"] = final_df["ligands"]+"_"+final_df["receptor"]
    # final_df = final_df.drop_duplicates()

    x = [
        "cluster7_tumors.csv",
        "cluster6_neutrophil.csv",
        "cluster5_myeloid.csv",
        "cluster4_mast.csv",
        "cluster3_lymphocyte.csv",
        "cluster2_fibroblast.csv",
        "cluster1_epithelial.csv",
        "cluster0_endothelial.csv",
    ]

    comb = [p for p in itertools.product(x, repeat=2)]

    # For each combination compute raw expression and cell cell interaction score
    for i in comb:
        # Read importance scores
        celltypeimp1 = pd.read_csv("../cluster_results_gradcam_gt1/lung13/" + i[0])

        if "_" not in str(celltypeimp1.iat[2, 1]):
            celltypeimp1["cellID"] = celltypeimp1["fov"].astype("str") + "_" + celltypeimp1["cellID"].astype("str")
        celltypeimp1.set_index("cellID", inplace=True)

        celltypeimp2 = pd.read_csv("../cluster_results_gradcam_gt1/lung13/" + i[1])

        if "_" not in str(celltypeimp2.iat[2, 1]):
            celltypeimp2["cellID"] = celltypeimp2["fov"].astype("str") + "_" + celltypeimp2["cellID"].astype("str")
        celltypeimp2.set_index("cellID", inplace=True)

        celltype1 = i[0].split("_")[1].split(".")[0]
        celltype2 = i[1].split("_")[1].split(".")[0]

        # Filter by the celltype
        final_df1 = final_df[(final_df["celltype1"] == celltype1) & (final_df["celltype2"] == celltype2)].copy()
        final_df1 = final_df1.drop_duplicates()

        # Compute the scores
        if len(final_df1) != 0:
            # Get raw expression and importance score for ligand and receptor genes
            def select_cell(row):
                row["raw_exp_cell1_ligands"] = exp.loc[row["Cell1"], row["ligands"]]
                row["raw_exp_cell2_receptor"] = exp.loc[row["Cell2"], row["receptor"]]
                row["raw_exp_cell2_ligands"] = exp.loc[row["Cell2"], row["ligands"]]
                row["raw_exp_cell1_receptor"] = exp.loc[row["Cell1"], row["receptor"]]
                row["score1"] = celltypeimp1.loc[row["Cell1"], row["ligands"]]
                row["score2"] = celltypeimp2.loc[row["Cell2"], row["receptor"]]
                row["score3"] = celltypeimp2.loc[row["Cell1"], row["ligands"]]
                row["score4"] = celltypeimp1.loc[row["Cell2"], row["receptor"]]

                return row

            final_df1 = final_df1.apply(lambda row: select_cell(row), axis=1)

            final_df1["score_1"] = final_df1.score1 * final_df1.score2
            final_df1["score_2"] = final_df1.score3 * final_df1.score4

            final_df1["raw_exp_score_1"] = final_df1.raw_exp_cell1_ligands * final_df1.raw_exp_cell2_receptor
            final_df1["raw_exp_score_2"] = final_df1.raw_exp_cell2_ligands * final_df1.raw_exp_cell1_receptor

            cci_results = pd.concat([cci_results, final_df1])

            # Group by ligand receptor pair
            score1 = final_df1.groupby("ligand-receptor")["score_1"].mean().reset_index()
            score2 = final_df1.groupby("ligand-receptor")["score_2"].mean().reset_index()

            lr_score1 = pd.concat([lr_score1, score1], axis=0)
            lr_score2 = pd.concat([lr_score2, score2], axis=0)

            # Append to list for pvalue computation
            scores1 = list(score1["score_1"])
            scores.extend(scores1)

            # Append to list for pvalue computation
            scores2 = list(score2["score_2"])
            scores.extend(scores2)

            # Append to list to store corresponding lr pairs
            pairs1.extend(list(score1["ligand-receptor"]) + list(score2["ligand-receptor"]))

            del score1
            del score2

        del celltypeimp1
        del celltypeimp2
    return lr_score1, lr_score2, cci_results


# Read ligand receptor pairs
lr = pd.read_csv("LRdb_saved_files.csv")

scores = []
pairs = []

# We need to save this data to file
cci_results = pd.DataFrame(
    columns=[
        "Cell1",
        "Cell2",
        "ligands",
        "receptor",
        "celltype1",
        "celltype2",
        "ligand-receptor",
        "score1",
        "score2",
        "score_1",
        "score3",
        "score4",
        "score_2",
    ]
)
lr_score1 = pd.DataFrame()
lr_score2 = pd.DataFrame()

# Computing seprately for small subset of lr pairs as having memory issues otherwise
lr_score1, lr_score2, cci_results = check(0, 500, scores, pairs, lr_score1, lr_score2, cci_results)
lr_score1, lr_score2, cci_results = check(500, 1000, scores, pairs, lr_score1, lr_score2, cci_results)
lr_score1, lr_score2, cci_results = check(1000, 1500, scores, pairs, lr_score1, lr_score2, cci_results)
lr_score1, lr_score2, cci_results = check(1500, 2000, scores, pairs, lr_score1, lr_score2, cci_results)
lr_score1, lr_score2, cci_results = check(2000, 2500, scores, pairs, lr_score1, lr_score2, cci_results)
lr_score1, lr_score2, cci_results = check(2500, 3500, scores, pairs, lr_score1, lr_score2, cci_results)

# Save the data to file for debugging
if not os.path.exists("./enrichment_results/"):
    os.makedirs("./enrichment_results/")
cci_results.to_csv("./enrichment_results/fov1617181920.csv")


# Statistical test
zscore = stats.zscore(scores)
pvalue = stats.norm.sf(abs(zscore))

fdr = multitest.fdrcorrection(pvalue)[1]
fdr_threshold = 0.05

fdr_columns = []
for i in range(len(fdr)):
    if fdr[i] <= fdr_threshold:
        fdr_columns.append(pairs[i])

# Print the significant pairs
for pair in fdr_columns:
    print(pair)
