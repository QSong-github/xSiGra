import argparse
import os
import random
import sys
from collections import Counter

import anndata
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torchvision import models, transforms

sys.path.append("../")
from xSiGra_model.transModel import ClusteringLayer, xSiGraModel, TransImg
from xSiGra_model.utils import (
    Cal_Spatial_Net,
    Stats_Spatial_Net,
    Transfer_img_Data,
    _hungarian_match,
)

cudnn.deterministic = True
cudnn.benchmark = False
plt.box(False)


# VGG feature extractor
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


# Initialize the model
model = models.vgg16(pretrained=True)
vgg_extractor = FeatureExtractor(model)

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.2/bin:$PATH"


def test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=[512, 30],
    random_seed=1234,
    device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/nanostring_train_lung13/all/",
    gt_mapping=None,
    cell_type=None,
):
    # Set seed
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

    # Pre-process
    datas = []
    labels = []
    gene_dim = 0
    img_dim = 0

    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)
    hidden_dims = opt.neurons
    arr = []

    model = models.vgg16(pretrained=True)
    vgg_extractor = FeatureExtractor(model)
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    vgg_extractor = vgg_extractor.to(device)

    for i in hidden_dims.split(","):
        arr.append(int(i))
    hidden_dims = arr

    for adata in adatas:
        import scipy.sparse as sp

        adata.X = sp.csr_matrix(adata.X)
        obsm_df = pd.DataFrame(adata.X.toarray(), columns=adata.to_df().columns)

        # Read explanations
        if cell_type == "tumors":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster7_tumors.csv")
        elif cell_type == "neutrophil":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster6_neutrophil.csv")
        elif cell_type == "myeloid":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster5_myeloid.csv")
        elif cell_type == "mast":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster4_mast.csv")
        elif cell_type == "lymphocyte":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster3_lymphocyte.csv")
        elif cell_type == "fibroblast":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster2_fibroblast.csv")
        elif cell_type == "epithelial":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster1_epithelial.csv")
        elif cell_type == "endothelial":
            gradient = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster0_endothelial.csv")

        # Select only gene scores
        markers = list(gradient.columns[10:])
        markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
        gradient = gradient[markers]

        # Top 30 genes sorted by variance in descending order
        genes = gradient.var().sort_values(ascending=False)[:30]
        df = genes.to_frame().reset_index()
        df = df.rename(columns={"index": "genes"})
        genes = df["genes"].tolist()
        genes = [x.split(" ")[0] for x in genes]
        index = [obsm_df.columns.get_loc(c) for c in genes]

        # Consider all genes
        adata.X = sp.csr_matrix(obsm_df.to_numpy())
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]

        # Transform the image, so it becomes readable with the model
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        features = []
        img1 = img.x.numpy()

        for i in range(img1.shape[0]):
            img = img1[i].reshape(w * 2, h * 2, 3)
            img = transform(np.uint8(img))
            img = img.reshape(1, 3, 224, 224)
            img = img.to(device)

            # Use pre-trained model
            with torch.no_grad():
                # Get features from image
                feature = vgg_extractor(img)

            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1))

        features = np.array(features)
        features = np.vstack(features).astype(np.float)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]
        labels = adata.obs["merge_cell_type"].tolist()

        le = preprocessing.LabelEncoder()
        gt = le.fit_transform(labels)
        y = pd.DataFrame(gt, columns=["merge_cell_type"])
        labels = y.replace({"merge_cell_type": gt_mapping})

        le = preprocessing.LabelEncoder()
        gt = le.fit_transform(labels)
        gt = np.transpose(gt)

        data.x = torch.cat([data.x, img, torch.tensor(np.asarray(gt)).view(len(gt), 1)], dim=1)
        datas.append(data)

    adata = anndata.concat(adatas)
    loader = DataLoader(datas, batch_size=1, num_workers=8, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    # Create clustering layer to detect cell type
    xplainmodel = xSiGraModel(
        hidden_dims=[gene_dim, img_dim] + hidden_dims,
        pretrained_model=model,
        num_clusters=2,
    ).to(device)

    if model_name is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(save_path, model_name),
                map_location=torch.device(opt.device),
            )
        )
        xplainmodel.load_state_dict(
            torch.load(
                os.path.join(save_path, model_name),
                map_location=torch.device(opt.device),
            ),
            strict=False,
        )
    else:
        print(os.path.join(save_path, opt.pretrain))

        # Load saved model
        xplainmodel.load_state_dict(
            torch.load(
                os.path.join(save_path, opt.pretrain),
                map_location=torch.device(opt.device),
            ),
            strict=False,
        )
        model.load_state_dict(
            torch.load(
                os.path.join(save_path, opt.pretrain),
                map_location=torch.device(opt.device),
            )
        )
    model.eval()

    # Add classification layer weights
    xplainmodel.out1.weight.data = linearmodel.out1.weight.data
    xplainmodel.out2.weight.data = linearmodel.out2.weight.data
    xplainmodel.out1.bias.data = linearmodel.out1.bias.data
    xplainmodel.out2.bias.data = linearmodel.out2.bias.data
    xplainmodel.to(device)

    # Freeze model
    params = xplainmodel.state_dict()

    for name, param in xplainmodel.named_parameters():
        param.requires_grad = False

    xplainmodel.out1.weight.requires_grad = False
    xplainmodel.out1.bias.requires_grad = False
    xplainmodel.out2.weight.requires_grad = False
    xplainmodel.out2.bias.requires_grad = False

    cluster_weights = {}
    gradient_weights = {}
    cluster_weights_gene = {}
    cluster_probabilities = {}

    for i in range(0, opt.ncluster):
        cluster_weights[i] = None

    for i in range(0, opt.ncluster):
        cluster_weights_gene[i] = None

    for i in range(0, opt.ncluster):
        gradient_weights[i] = None

    for i in range(0, opt.ncluster):
        cluster_probabilities[i] = None

    outlabel = []
    gtlabel = []
    prob = []

    train_acc = []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim : gene_dim + 4096]
        labels = batch.x[:, gene_dim + 4096 :]

        edge_index = batch.edge_index
        bgene.requires_grad = True
        bgene = bgene.float()
        bimg = bimg.float()
        out = xplainmodel(bgene.float(), bimg.float(), edge_index, batch.batch)

        # Get model predictions
        x = torch.nn.functional.log_softmax(out, dim=1)
        prob.extend(x[:, 1].detach().cpu().tolist())
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.detach().cpu().tolist())

        labels = labels.tolist()
        labels = [int(x[0]) for x in labels]
        gtlabel.extend(labels)

    # Compute metrics
    print("Train accuracy:", accuracy_score(gtlabel, outlabel))
    train_acc = accuracy_score(gtlabel, outlabel)

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    auc = roc_auc_score(gtlabel, prob)
    print("AUC:", auc)

    return train_acc, score, auc


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


# Pre-process data
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
# Use our provided saved results for Lung 13 with best resolution or use your computed results and resolution (change)
adata_pred = sc.read("../saved_adata/" + opt.dataset + "_adata_pred.h5ad")
res = 0.3681250000000001

# res = res_search(adata_pred, opt.ncluster, opt.seed)
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

# leiden to ground truth cell type matching
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

# Replace leiden with matched groundtruth
obs_df = obs_df.replace({"leiden": dict_mapping})
adata_pred.obs["leiden"] = obs_df["leiden"]

ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
print("ARI: %.2f" % ARI)

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
dict1 = {}
dict_name = {}

dict_gtname = {}
for i in gt:
    dict_gtname[i] = le.classes_[i]
label = list(sorted(dict_gtname.values()))

dict_gtname = {}
for i in match:
    dict1[str(i[0])] = i[1]
    dict_name[i[0]] = le.classes_[i[1]]
    dict_gtname[i[1]] = le.classes_[i[0]]

obs_df = obs_df.replace({"leiden": dict1})
adata_pred.obs["leiden"] = obs_df["leiden"]
for val in set(adata_pred.obs["leiden"]):
    dict_name[val] = le.classes_[val]
label = label + list(dict_name.values())

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


obs_df = adata_pred.obs.dropna()
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
linearmodel = ClusteringLayer(2).to(device)

labels = obs_df["merge_cell_type"].tolist()

le = preprocessing.LabelEncoder()
gt = le.fit_transform(labels)
y = pd.DataFrame(gt, columns=["merge_cell_type"])
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])

# Store results
result_df = pd.DataFrame()
accuracies = []
celltypes = []
scores = []
aucs = []

# For tumor cluster
gradient_tumors = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster7_tumors.csv")
print("Tumors")
markers = list(gradient_tumors.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_tumors = gradient_tumors[markers]

# From tumor cluster select top 30 genes based on highest importance score variance
genes = gradient_tumors.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()

index = [gradient_tumors.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 1, 3: 0, 2: 0, 4: 0, 6: 0, 5: 0, 0: 0, 1: 0}
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

# Split in train and test
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
cross_el = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))

if not os.path.exists("./saved_model/" + opt.dataset + "_" + opt.benchmark):
    os.makedirs("./saved_model/" + opt.dataset + "_" + opt.benchmark)
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="tumors",
)

# Save results for tumor cluster
celltypes.append("tumors")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For fibroblast cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_fibroblast = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster2_fibroblast.csv")

print("Fibroblast")
markers = list(gradient_fibroblast.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_fibroblast = gradient_fibroblast[markers]

# From fibroblast cluster select top 30 genes based on highest importance score variance
genes = gradient_fibroblast.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_fibroblast.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 1, 4: 0, 6: 0, 5: 0, 0: 0, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

# Split in train and test
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="fibroblast",
)

# Save results for fibroblast cluster
celltypes.append("fibroblast")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For lymphocyte cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_lymphocyte = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster3_lymphocyte.csv")

print("Lymphocyte")
markers = list(gradient_lymphocyte.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_lymphocyte = gradient_lymphocyte[markers]

# From lymphocyte cluster select top 30 genes based on highest importance score variance
genes = gradient_lymphocyte.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_lymphocyte.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 1, 2: 0, 4: 0, 6: 0, 5: 0, 0: 0, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

# Split in train and test
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="lymphocyte",
)

# Save results for lymphocyte cluster
celltypes.append("lymphocyte")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For myeloid cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_myeloid = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster5_myeloid.csv")

print("Myeloid")
markers = list(gradient_myeloid.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_myeloid = gradient_myeloid[markers]

# From myeloid cluster select top 30 genes based on highest importance score variance
genes = gradient_myeloid.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_myeloid.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 0, 4: 0, 6: 0, 5: 1, 0: 0, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

# Split in train and test
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="myeloid",
)

# Save results for myeloid cluster
celltypes.append("myeloid")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For mast cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_mast = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster4_mast.csv")

print("Mast")
markers = list(gradient_mast.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_mast = gradient_mast[markers]

# From mast cluster select top 30 genes based on highest importance score variance
genes = gradient_mast.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_mast.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 0, 4: 1, 6: 0, 5: 0, 0: 0, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="mast",
)

# Save results for mast cluster
celltypes.append("mast")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For neutrophil cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_neutrophil = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster6_neutrophil.csv")

print("Neutrophil")
markers = list(gradient_neutrophil.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_neutrophil = gradient_neutrophil[markers]

# From neutrophil cluster select top 30 genes based on highest importance score variance
genes = gradient_neutrophil.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_neutrophil.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 0, 4: 0, 6: 1, 5: 0, 0: 0, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    print(Counter(outlabel))
    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="neutrophil",
)

# Save results for neutrophil cluster
celltypes.append("neutrophil")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For endothelial cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_endothelial = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster0_endothelial.csv")

print("Endothelial")
markers = list(gradient_endothelial.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_endothelial = gradient_endothelial[markers]

# From endothelial cluster select top 30 genes based on highest importance score variance
genes = gradient_endothelial.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_endothelial.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 0, 4: 0, 6: 0, 5: 0, 0: 1, 1: 0}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    from collections import Counter

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="endothelial",
)

# Save results for endothelial cluster
celltypes.append("endothelial")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

# For epithelial cluster
obsm_df = pd.DataFrame(adata_pred.obsm["pred"])
gradient_epithelial = pd.read_csv("../cluster_results_" + opt.benchmark + "_gt1/" + opt.dataset + "/cluster1_epithelial.csv")

print("Epithelial")
markers = list(gradient_epithelial.columns[10:])
markers = list(filter(lambda x: any("imp score" in x for x in markers), markers))
gradient_epithelial = gradient_epithelial[markers]

# From epithelial cluster select top 30 genes based on highest importance score variance
genes = gradient_epithelial.var().sort_values(ascending=False)[:30]
df = genes.to_frame().reset_index()
df = df.rename(columns={"index": "genes"})
genes = df["genes"].tolist()
index = [gradient_epithelial.columns.get_loc(c) for c in genes]

# Convert ground truth to binary where tumor cells are class 1 and others 0
gt_mapping = {7: 0, 3: 0, 2: 0, 4: 0, 6: 0, 5: 0, 0: 0, 1: 1}
y = pd.DataFrame(gt, columns=["merge_cell_type"])
y = y.replace({"merge_cell_type": gt_mapping})

# Create dataset
dataset = Custom_Dataset(obsm_df.to_numpy(), y["merge_cell_type"])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Oversampling due to imbalanced labels
oversample = SMOTE()
X = obsm_df.iloc[train_indices]
y = y["merge_cell_type"].iloc[train_indices]
X, y = oversample.fit_resample(X, y)
train_dataset = Custom_Dataset(X.to_numpy(), y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Data loader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(linearmodel.parameters(), lr=0.001, capturable=True)
epochs = 10

# Train model
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(epochs):
    linearmodel.train()
    correct = 0
    running_loss = 0.0
    total = 0
    gtlabel = []
    outlabel = []
    for idx, batch in enumerate(train_loader):
        data, labels = batch

        gtlabel.extend(labels.tolist())
        optimizer.zero_grad()
        out = linearmodel(data.float().to(device))
        loss = cross_el(out, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        x = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(x, dim=1)
        outlabel.extend(pred.cpu().tolist())
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")
    gtlabel = np.array(gtlabel).flatten()
    outlabel = np.array(outlabel).flatten()

    print("Train accuracy:", accuracy_score(gtlabel, outlabel))

    score = f1_score(outlabel, gtlabel, average="weighted")
    print("F1 score train:", score)

    print("AUC:", roc_auc_score(gtlabel, outlabel))
torch.save(
    linearmodel.state_dict(),
    "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
)

# Test model
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(2).to(device)
linearmodel.load_state_dict(
    torch.load(
        "./saved_model/" + opt.dataset + "_" + opt.benchmark + "/linearmodel_gt_all.pth",
        map_location=torch.device(opt.device),
    )
)
linearmodel.eval()

gtlabel = []
outlabel = []

for idx, batch in enumerate(test_loader):
    data, labels = batch
    gtlabel.extend(labels.tolist())
    out = linearmodel(data.float().to(device))

    x = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(x, dim=1)
    outlabel.extend(pred.cpu().tolist())
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")
gtlabel = np.array(gtlabel).ravel()
outlabel = np.array(outlabel).ravel()

print("Test accuracy:", accuracy_score(gtlabel, outlabel))

score = f1_score(outlabel, gtlabel, average="weighted")
print("F1 score test:", score)

print("AUC:", roc_auc_score(gtlabel, outlabel))

print("Explanations")

# Compute fidelity
acc, score, auc = test_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
    gt_mapping=gt_mapping,
    cell_type="epithelial",
)

# Save results for epithelial cluster
celltypes.append("epithelial")
accuracies.append(acc)
scores.append(score)
aucs.append(auc)

if not os.path.exists("./fidelity_results/"):
    os.makedirs("./fidelity_results/")
results_df = pd.DataFrame.from_dict({"celltype": celltypes, "accuracy": accuracies, "F1 score": scores, "auc": aucs})
results_df.to_csv("./fidelity_results/fidelity_all_" + opt.benchmark + "_" + opt.dataset)
