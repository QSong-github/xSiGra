import argparse
import os
import random
import sys
import time

import anndata
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torchvision import models, transforms

sys.path.append("../")
from xSiGra_model.transModel import ClusteringLayer, xSiGraModel, TransImg
from xSiGra_model.utils import Cal_Spatial_Net, Stats_Spatial_Net, Transfer_img_Data

plt.box(False)

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--root", type=str, default="../dataset/nanostring/lung13")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--id", type=str, default="fov1")
parser.add_argument("--img_name", type=str, default="F001")
parser.add_argument("--neurons", type=str, default="512,30")
parser.add_argument("--num_layers", type=str, default="2")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_path", type=str, default="../checkpoint/nanostring_train_lung13")
parser.add_argument("--ncluster", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--use_gray", type=float, default=0)
parser.add_argument("--test_only", type=int, default=1)
parser.add_argument("--pretrain", type=str, default="best.pth")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--img_size", type=str, default="60,60")
parser.add_argument("--cluster_method", type=str, default="leiden")
parser.add_argument("--num_fov", type=int, default=20)
parser.add_argument("--dataset", type=str, default="lung13")
parser.add_argument("--benchmark", type=str, default="deconvolution")

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

    print(os.path.join(root, id, "CellComposite_%s.jpg" % (img_name)))
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

adatas = list()
for id, name in zip(ids, img_names):
    adata = gen_adatas(opt.root, id, name)
    adatas.append(adata)

sp = os.path.join(opt.save_path, "all")

# Uncomment to select model from scratch
# Choose best model
# adata_pred = test_nano_fov(
#     opt,
#     adatas,
#     hidden_dims=opt.neurons,
#     random_seed=opt.seed,
#     save_path=sp,
# )

# sc.pp.neighbors(adata_pred, opt.ncluster, use_rep='pred')

# def res_search(adata_pred, ncluster, seed, iter=200):
#     start = 0
#     end = 3
#     i = 0
#     while start < end:
#         if i >= iter:
#             return res
#         i += 1
#         res = (start + end) / 2
#         print(res)
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
#             print("find", res)
#             return res
#         if count > ncluster:
#             end = res
#         else:
#             start = res
#     raise NotImplementedError()
# res = res_search(adata_pred, opt.ncluster, opt.seed)

# Comment below lines if selecting model above
# Use our provided saved results for Lung 13 with best resolution or use your computed results and resolution (change)
adata_pred = sc.read("../saved_adata/" + opt.dataset + "_adata_pred.h5ad")
res = 0.3681250000000001

sc.tl.leiden(adata_pred, resolution=res, key_added="leiden", random_state=opt.seed)
obs_df = adata_pred.obs.dropna()

# Compute ARI
ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)

ARI2 = adjusted_rand_score(obs_df["merge_cell_type"], obs_df["leiden"])
print("ARI2: %.2f" % ARI2)

labels = obs_df["merge_cell_type"]


# Match clusters to cell types
def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_target.shape[0]
    num_k = preds_k
    num_k_tar = target_k
    num_correct = np.zeros((num_k, num_k_tar))
    for c1 in range(num_k):
        for c2 in range(num_k_tar):
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

label = list(sorted(dict_gtname.values()))

dict_gtname = {}
for i in match:
    dict_mapping[str(i[0])] = i[1]
    dict_name[i[0]] = le.classes_[i[1]]
    dict_gtname[i[1]] = le.classes_[i[0]]
print(dict_gtname)
obs_df = obs_df.replace({"leiden": dict_mapping})
adata_pred.obs["leiden"] = obs_df["leiden"]

ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)


# Train linear layer to convert leiden clusters to probability
class Custom_Dataset(Dataset):
    def __init__(self, indata, label):
        self.indata = indata
        self.label = [int(item) for item in label.values.tolist()]
        self.label = np.array(self.label)

    def __len__(self):
        return len(self.indata)

    def __getitem__(self, index):
        indata = torch.tensor(self.indata[index])
        label = torch.tensor(self.label[index])
        return indata, label


cross_el = torch.nn.CrossEntropyLoss()
obs_df = adata_pred.obs.dropna()
y = obs_df["leiden"].to_frame().reset_index()

device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
linearmodel = ClusteringLayer(opt.ncluster).to(device)

labels = obs_df["merge_cell_type"].tolist()
le = preprocessing.LabelEncoder()
gt = le.fit_transform(labels)
y = pd.DataFrame(gt, columns=["merge_cell_type"])

dataset = Custom_Dataset(adata_pred.obsm["pred"], y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001)
epochs = 30

train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0

    for idx, batch in enumerate(train_loader):
        data, labels = batch
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)
        if (idx) % 20 == 0:
            print(loss.item())
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")

torch.save(linearmodel.state_dict(), "../saved_model/" + opt.dataset + "/linearmodel_gt.pth")

# Testing
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(opt.ncluster).to(device)
linearmodel.load_state_dict(torch.load("../saved_model/" + opt.dataset + "/linearmodel_gt.pth"))
linearmodel = linearmodel.to(device)
linearmodel.eval()

for idx, batch in enumerate(test_loader):
    data, labels = batch
    out = linearmodel(data.float().to(device))
    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")

cudnn.deterministic = True
cudnn.benchmark = False
model = models.vgg16(pretrained=True)


# Pretrained VGG-16
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = model.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


model = models.vgg16(pretrained=True)
vgg_model = FeatureExtractor(model)

# Set R path
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.2/bin:$PATH"


def compute_explanations(gradients):
    gene_importance = gradients
    cell_importance = []
    cell_importance = F.relu((gene_importance).sum(dim=1, keepdims=True))
    cell_importance = (gene_importance).sum(dim=1, keepdims=True)

    return cell_importance, gene_importance


def get_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=[512, 30],
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/nanostring_train_lung13",
):
    start_time = time.time()
    seed = random_seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt.pretrain = "best.pth"
    datas = []
    gene_dim = 0
    img_dim = 0

    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)
    hidden_dims = opt.neurons

    arr = []

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True)

    # Load VGG model
    vgg_model = FeatureExtractor(model)
    vgg_model = vgg_model.to(device)

    for i in hidden_dims.split(","):
        arr.append(int(i))
    hidden_dims = arr

    # Transform data
    for adata in adatas:
        import scipy.sparse as sp

        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        features = []
        img_transformed = img.x.numpy()

        for i in range(img_transformed.shape[0]):
            img = img_transformed[i].reshape(w * 2, h * 2, 3)
            img = transform(np.uint8(img))
            img = img.reshape(1, 3, 224, 224)
            img = img.to(device)

            with torch.no_grad():
                feature = vgg_model(img)

            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1))

        features = np.array(features)
        features = np.vstack(features).astype(np.float)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]

        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)

    adata = anndata.concat(adatas)
    loader = DataLoader(datas, batch_size=1, num_workers=8, shuffle=False)

    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    if model_name is not None:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        xplainmodel.load_state_dict(torch.load(os.path.join(save_path, model_name)), strict=False)
    else:
        model.load_state_dict(
            torch.load(
                os.path.join(save_path, opt.pretrain),
                map_location=torch.device(opt.device),
            )
        )

    model.eval()
    xplainmodel = xSiGraModel(
        hidden_dims=[gene_dim, img_dim] + hidden_dims,
        pretrained_model=model,
        num_clusters=opt.ncluster,
    ).to(device)

    xplainmodel.load_state_dict(
        torch.load(os.path.join(save_path, opt.pretrain), map_location=device),
        strict=False,
    )

    # Load the linear layer weights
    xplainmodel.out1.weight.data = linearmodel.out1.weight.data.to(device)
    xplainmodel.out2.weight.data = linearmodel.out2.weight.data.to(device)
    xplainmodel.out1.bias.data = linearmodel.out1.bias.data.to(device)
    xplainmodel.out2.bias.data = linearmodel.out2.bias.data.to(device)

    # pretrained_dict = model.state_dict()
    # model_dict = xplainmodel.state_dict()

    # processed_dict = {}

    # for k in model_dict.keys():
    #     decomposed_key = k.split(".")
    #     if "model" in decomposed_key:
    #         pretrained_key = ".".join(decomposed_key[1:])
    #         processed_dict[k] = pretrained_dict[pretrained_key]
    # xplainmodel.load_state_dict(processed_dict, strict=False)

    # Freeze model
    for name, param in xplainmodel.named_parameters():
        param.requires_grad = False

    xplainmodel.out1.weight.requires_grad = False
    xplainmodel.out1.bias.requires_grad = False
    xplainmodel.out2.weight.requires_grad = False
    xplainmodel.out2.bias.requires_grad = False

    cluster_weights_cell = {}
    cluster_weights_gene = {}

    for i in range(0, opt.ncluster):
        cluster_weights_cell[i] = None

    for i in range(0, opt.ncluster):
        cluster_weights_gene[i] = None

    # Stores the cluster label for each cell
    outlabel = []
    end_time = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        edge_index = batch.edge_index

        bgene = bgene.float()
        bgene.requires_grad = True
        bimg = bimg.float()
        clusterlabel = xplainmodel(bgene.float(), bimg.float(), edge_index, batch.batch)

        for label in clusterlabel:
            _, pred = torch.max(label, dim=0)
            outlabel.append(pred.item())

        # For each cluster get node and gene importance scores
        for k in range(0, opt.ncluster):
            if opt.benchmark == "deconvolution":
                from captum.attr import Deconvolution

                deconv = Deconvolution(xplainmodel)
                gradients = deconv.attribute(
                    bgene.float(),
                    target=k,
                    additional_forward_args=(bimg.float(), edge_index, batch.batch),
                )
                cell_explain, gene_explain = compute_explanations(gradients)
            elif opt.benchmark == "guidedbackprop":
                from captum.attr import GuidedBackprop

                gp = GuidedBackprop(xplainmodel)
                gradients = gp.attribute(
                    bgene.float(),
                    target=k,
                    additional_forward_args=(bimg.float(), edge_index, batch.batch),
                )
                cell_explain, gene_explain = compute_explanations(gradients)
            elif opt.benchmark == "saliency":
                from captum.attr import Saliency

                sal = Saliency(xplainmodel)
                gradients = sal.attribute(
                    bgene.float(),
                    target=k,
                    additional_forward_args=(bimg.float(), edge_index, batch.batch),
                )
                cell_explain, gene_explain = compute_explanations(gradients)
            elif opt.benchmark == "inputxgradient":
                from captum.attr import InputXGradient

                inputx = InputXGradient(xplainmodel)
                gradients = inputx.attribute(
                    bgene.float(),
                    target=k,
                    additional_forward_args=(bimg.float(), edge_index, batch.batch),
                )
                cell_explain, gene_explain = compute_explanations(gradients)

            if cluster_weights_cell[k] is None:
                cluster_weights_cell[k] = cell_explain
            else:
                cluster_weights_cell[k] = torch.cat((cluster_weights_cell[k], cell_explain))

            if cluster_weights_gene[k] is None:
                cluster_weights_gene[k] = gene_explain
            else:
                cluster_weights_gene[k] = torch.cat((cluster_weights_gene[k], gene_explain))

    labels = adata.obs["merge_cell_type"]

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

    leiden = np.array(outlabel)
    match = _hungarian_match(leiden.astype(np.int8), gt.astype(np.int8), opt.ncluster, 8)

    # Maps cluster number to cell type using humgarian matching
    dict_cluster_map_cell = {}
    for mapping in match:
        dict_cluster_map_cell[mapping[0]] = le.classes_[mapping[1]]

    if opt.benchmark == "inputxgradient":
        for i in range(0, opt.ncluster):
            adata.obs["cmap" + str(i)] = np.array(cluster_weights_cell[i].detach().cpu())
    else:
        for i in range(0, opt.ncluster):
            adata.obs["cmap" + str(i)] = np.array(cluster_weights_cell[i].cpu())
    if opt.benchmark == "inputxgradient":
        for i in range(0, opt.ncluster):
            adata.uns["gmap" + str(i)] = np.array(cluster_weights_gene[i].detach().cpu())
    else:
        for i in range(0, opt.ncluster):
            adata.uns["gmap" + str(i)] = np.array(cluster_weights_gene[i].cpu())

    keys_list = ["cmap" + str(i) for i in range(opt.ncluster)]
    genedf = sc.get.obs_df(adata, keys=keys_list)

    count = 0
    start = 0

    ids = ["fov" + str(i) for i in range(1, int(opt.num_fov) + 1)]
    img_names = ["F00" + str(i) for i in range(1, 10)]
    img_names = img_names + ["F0" + str(i) for i in range(10, int(opt.num_fov) + 1)]

    datas = []
    adata_pred = adata

    all_cells_df = pd.DataFrame()
    fov = 0
    fovs = []
    dict_cmap = {}
    gt = []
    for i in range(0, opt.ncluster):
        dict_cmap["cmap" + str(i)] = []

    # For each FOV
    for id in ids:
        fov += 1

        adata = sc.read(os.path.join("../dataset/nanostring/" + opt.dataset + "/", id, "sampledata.h5ad"))
        start = count
        end = start + adata.shape[0]
        count = count + adata.shape[0]

        adata.var_names_make_unique()
        gt = gt + adata.obs["merge_cell_type"].tolist()

        for i in range(opt.ncluster):
            adata.uns["gmap" + str(i)] = adata_pred.uns["gmap" + str(i)]
            adata.obs["cmap" + str(i)] = genedf.iloc[start:end]["cmap" + str(i)]
            dict_cmap["cmap" + str(i)] = dict_cmap["cmap" + str(i)] + adata.obs["cmap" + str(i)].tolist()

        all_cells_df1 = adata.to_df()
        all_cells_df = pd.concat([all_cells_df, all_cells_df1])
        fovs = fovs + [fov for i in range(len(all_cells_df1))]

        # # Plot cell importance for each fov
        for k in range(0, opt.ncluster):
            adata.obs["cmap" + str(k)] = genedf.iloc[start:end]["cmap" + str(k)]

    gene_names = list(all_cells_df.columns)
    adata = adata_pred

    # For each cluster create csv file
    ## Fov
    ## CellID
    ## CellType
    ## Cell importance score
    ## Gene importance score along with pvalue, zscore, pval_adjusted_fdr and pval_adjusted_bonferroni

    for k in range(0, opt.ncluster):
        cell_id = all_cells_df.reset_index()["cell_ID"]

        gmap = adata.uns["gmap" + str(k)]
        gmap_df = pd.DataFrame(gmap, columns=gene_names)
        genes_df = pd.concat([cell_id, gmap_df], axis=1)

        # Sort genes with highest variance
        top_genes = genes_df.var().sort_values(ascending=False)
        top_genes_df = top_genes.to_frame().reset_index()
        top_genes_df = top_genes_df.rename(columns={"index": "genes"})
        genes = top_genes_df["genes"].tolist()

        cluster_metadata_df = pd.DataFrame()
        cluster_metadata_df["fov"] = fovs
        cluster_metadata_df["cellID"] = cell_id
        cluster_metadata_df["cell_type"] = gt
        cluster_metadata_df["cell_imp_score"] = dict_cmap["cmap" + str(k)]

        cluster_metadata_df["cluster"] = outlabel
        cluster_metadata_df["cx"] = adata.obs["cx"].tolist()
        cluster_metadata_df["cy"] = adata.obs["cy"].tolist()
        cluster_metadata_df["cx_g"] = adata.obs["cx_g"].tolist()
        cluster_metadata_df["cy_g"] = adata.obs["cy_g"].tolist()

        for gene in genes:
            cluster_metadata_df[gene + " imp score"] = genes_df[gene]

        if not os.path.exists(os.path.join("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset)):
            os.makedirs(os.path.join("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset))
        cluster_metadata_df.to_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster" + str(k) + "_" + dict_cluster_map_cell[k] + ".csv")

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print("Downstream task time in hours:", hours)
    return


# Get explanatory results
get_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=1234,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
)
