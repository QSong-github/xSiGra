import argparse
import os
import random

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
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from train_transformer import get_explanations, test_nano_fov
from transModel import ClusteringLayer
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match

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
label = list(sorted(dict_gtname.values()))

dict_gtname = {}
for i in match:
    dict_mapping[str(i[0])] = i[1]
    dict_name[i[0]] = le.classes_[i[1]]
    dict_gtname[i[1]] = le.classes_[i[0]]

obs_df = obs_df.replace({"leiden": dict_mapping})
adata_pred.obs["leiden"] = obs_df["leiden"]


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
        probs = torch.nn.functional.log_softmax(out, dim=1)
        _, pred = torch.max(probs, dim=1)
        correct += torch.sum(pred == labels.to(device)).item()
        total += labels.size(0)
        if (idx) % 20 == 0:
            print(loss.item())
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")

if not os.path.exists(os.path.join("../saved_model/" + opt.dataset)):
    os.makedirs(os.path.join("../saved_model/" + opt.dataset))
torch.save(linearmodel.state_dict(), "../saved_model/" + opt.dataset + "/linearmodel_gt.pth")

# Testing
correct = 0
running_loss = 0.0
total = 0
total_step = len(test_loader)

linearmodel = ClusteringLayer(opt.ncluster).to(device)
linearmodel.load_state_dict(torch.load("../saved_model/" + opt.dataset + "/linearmodel_gt.pth"))
linearmodel.eval()

for idx, batch in enumerate(test_loader):
    data, labels = batch
    out = linearmodel(data.float().to(device))
    probs = torch.nn.functional.log_softmax(out, dim=1)
    _, pred = torch.max(probs, dim=1)
    correct += torch.sum(pred == labels.to(device)).item()
    total += labels.size(0)

test_acc = 100 * correct / total
print(f"\ntest acc: {(100 * correct / total):.4f}")

# Get explanatory results for different clusters
get_explanations(
    opt,
    adatas,
    linearmodel,
    model_name=None,
    hidden_dims=opt.neurons,
    random_seed=0,
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu"),
    save_path=opt.save_path + "/all/",
)
