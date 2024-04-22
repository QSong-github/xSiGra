import os

import anndata as AD
import cv2
import numpy as np
import pandas as pd


# Function for preprocessing
def gen_h5ad(id, img_id, fov):
    # Read gene expression and image files
    root = "../dataset/nanostring/lung13/"
    img_root = os.path.join(root, id, "CellComposite_%s.jpg" % (img_id))
    img = cv2.imread(img_root)
    height, width, c = img.shape
    gene_expression = os.path.join(root, "Lung13_exprMat_file.csv")
    ge = pd.read_csv(gene_expression, delimiter=",")

    # Filter gene expression
    gene_f1 = ge[ge["fov"] == int(fov)]
    gene_f1 = gene_f1.drop(columns=["fov"])
    gene_f1 = gene_f1.set_index("cell_ID")
    idx = gene_f1.index

    # Read annotation
    annor = os.path.join(root, "matched_annotation_lung13.csv")
    anno = pd.read_csv(annor)
    anno_f1 = anno[anno["fov"] == int(fov)]

    # Compute image patch and match to cell centre
    w, h = 60, 60

    for i, row in anno_f1.iterrows():
        cx, cy = float(anno_f1["CenterX_local_px"][i]), float(anno_f1["CenterY_local_px"][i])
        anno_f1["CenterY_local_px"][i] = height - float(anno_f1["CenterY_local_px"][i])

        if cx - w < 0 or cx + w > width or cy - h < 0 or cy + h > height:
            anno_f1["cell_type"][i] = np.nan

    anno_f1 = anno_f1.set_index("cell_ID").reindex(idx)

    # Drop corresponding rows in gene expression if annotation is nan
    gene_f1["cell_type"] = anno_f1["cell_type"]
    # gene_f1['niche'] = anno_f1['niche']
    gene_f1 = gene_f1.dropna(axis=0, how="any")
    gene_f1 = gene_f1.drop(columns=["cell_type"])

    # build anndata
    adata = AD.AnnData(gene_f1)
    anno_f1.index = anno_f1.index.map(str)

    adata.obs["cell_type"] = anno_f1.loc[adata.obs_names, "cell_type"]
    # adata.obs['niche'] = anno_f1.loc[adata.obs_names, 'niche']

    adata.obs["cx"] = anno_f1.loc[adata.obs_names, "CenterX_local_px"]
    adata.obs["cy"] = anno_f1.loc[adata.obs_names, "CenterY_local_px"]

    adata.obs["cx_g"] = anno_f1.loc[adata.obs_names, "CenterX_global_px"]
    adata.obs["cy_g"] = anno_f1.loc[adata.obs_names, "CenterY_global_px"]

    df = pd.DataFrame(index=adata.obs.index)
    df["cx"] = adata.obs["cx"]
    df["cy"] = adata.obs["cy"]
    arr = df.to_numpy()
    adata.obsm["spatial"] = arr

    df = pd.DataFrame(index=adata.obs.index)
    df["cx_g"] = adata.obs["cx_g"]
    df["cy_g"] = adata.obs["cy_g"]
    arr = df.to_numpy()

    adata.obsm["spatial_global"] = arr

    # Merge cell types

    dicts = {}

    dicts["T CD8 memory"] = "lymphocyte"
    dicts["T CD8 naive"] = "lymphocyte"
    dicts["T CD4 naive"] = "lymphocyte"
    dicts["T CD4 memory"] = "lymphocyte"
    dicts["Treg"] = "lymphocyte"
    dicts["B-cell"] = "lymphocyte"
    dicts["plasmablast"] = "lymphocyte"
    dicts["NK"] = "lymphocyte"
    dicts["monocyte"] = "myeloid"
    dicts["macrophage"] = "myeloid"
    dicts["mDC"] = "myeloid"
    dicts["pDC"] = "myeloid"
    dicts["tumors"] = "tumors"
    dicts["myeloid"] = "myeloid"
    dicts["lymphocyte"] = "lymphocyte"
    dicts["epithelial"] = "epithelial"
    dicts["mast"] = "mast"
    dicts["endothelial"] = "endothelial"
    dicts["fibroblast"] = "fibroblast"
    dicts["neutrophil"] = "neutrophil"

    adata.obs["merge_cell_type"] = np.zeros(adata.shape[0])
    for key, v in dicts.items():
        idx = adata.obs["cell_type"] == key
        adata.obs["merge_cell_type"][idx] = v

    # Save anndata
    # adata.obs["merge_cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs.columns = adata.obs.columns.astype(str)
    adata.var.columns = adata.var.columns.astype(str)
    adata.write(os.path.join(root, id, "sampledata.h5ad"))


def processing_nano():
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

    fov = 1
    for id, imname in zip(ids, img_names):
        gen_h5ad(id, imname, fov)
        fov += 1


if __name__ == "__main__":
    processing_nano()
