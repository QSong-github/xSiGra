import os
import random
import sys
import time

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from torch import nn
from torch_geometric.loader import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

sys.path.append("../")
from transModel import xSiGraModel, TransImg
from utils import Transfer_img_Data

cudnn.deterministic = True
cudnn.benchmark = False

plt.box(False)
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)


# Feature Extractor
# Update to include other pre-trained models
class FeatureExtractor(nn.Module):
    def __init__(self, model, model_type='vgg'):
        super(FeatureExtractor, self).__init__()
        self.model_type = model_type
        if model_type == 'vgg':
            self.features = nn.Sequential(*list(model.features))
            self.pooling = model.avgpool
            self.flatten = nn.Flatten()
            self.fc = model.classifier[0]
        elif model_type == 'resnet':
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.flatten = nn.Flatten()
            self.fc = model.fc

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out) if self.model_type == 'vgg' else out
        out = self.flatten(out)
        out = self.fc(out)
        return out

vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Feature extractors
vgg_extractor = FeatureExtractor(vgg_model, model_type='vgg')
resnet_extractor = FeatureExtractor(resnet_model, model_type='resnet')

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.2/bin:$PATH"


@torch.no_grad()
def test_nano_fov(
    opt,
    adatas,
    model_name=None,
    hidden_dims=[512, 30],
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/nanostring_lung13",
):
    start_time = time.time()

    # Set seed
    seed = random_seed

    # Get VGG instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    vgg_extractor = vgg_extractor.to(device)

    # Pre-process data
    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)

    hidden_dims = opt.neurons
    hidden_dims_arr = hidden_dims.split(",")
    arr = []
    for i in hidden_dims_arr:
        arr.append(int(i))
    hidden_dims = arr

    # Set seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datas = []
    gene_dim = 0
    img_dim = 0

    all_cells_df = pd.DataFrame()
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        adata_df = adata.to_df()
        all_cells_df = pd.concat([all_cells_df, adata_df])

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

        features = []
        img_transformed = img.x.numpy()

        for i in range(img_transformed.shape[0]):
            img = img_transformed[i].reshape(3, w * 2, h * 2)
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img))   
            img = img.unsqueeze(0)
            img = img.to(device)
            with torch.no_grad():
                feature = vgg_extractor(img)

            img = feature
            features.append(feature.cpu().detach().numpy().reshape(-1))

        features = np.array(features)
        features = np.vstack(features).astype(np.float64)
        img = torch.from_numpy(features)
        img_dim = features.shape[1]

        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)

    adata = anndata.concat(adatas)

    # Data loader
    loader = DataLoader(datas, batch_size=1, num_workers=0, shuffle=False)

    # Create model instance
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    best_davies = float("inf")
    best_adata = None

    # Select best model using Davies Bouldin score amoung the last 10 saved
    for k in range(191, 201):
        opt.pretrain = "final_" + str(k) + "_0.pth"

        if model_name is not None:
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        else:
            print(os.path.join(save_path, opt.pretrain))
            model.load_state_dict(
                torch.load(
                    os.path.join(save_path, opt.pretrain),
                    map_location=torch.device(opt.device),
                )
            )

        hidden_matrix = None
        gene_matrix = None
        img_matrix = None
        couts = None

        # Get model predictions
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]

            bgene = bgene.float()
            bimg = bimg.float()
            bgene.requires_grad = True
            edge_index = batch.edge_index
            gz, iz, cz, gout, iout, cout, gap = model(bgene, bimg, edge_index, batch.batch)

            if hidden_matrix is None:
                hidden_matrix = cz.detach().cpu()
                gene_matrix = gz.detach().cpu()
                couts = cout.detach().cpu()
                img_matrix = iz.detach().cpu()
            else:
                hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
                gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
                img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
                couts = torch.cat([couts, cout.detach().cpu()], dim=0)

        hidden_matrix = hidden_matrix.numpy()
        gene_matrix = gene_matrix.numpy()
        img_matrix = img_matrix.numpy()
        adata.obsm["pred"] = hidden_matrix
        adata.obsm["gene_pred"] = gene_matrix
        adata.obsm["img_pred"] = img_matrix
        couts = couts.numpy().astype(np.float64)
        couts[couts < 0] = 0
        adata.layers["recon"] = couts
        adata_pred = adata

        # Perform leiden clustering
        sc.pp.neighbors(adata_pred, opt.ncluster, use_rep="pred")

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

        res = res_search(adata_pred, opt.ncluster, opt.seed)
        sc.tl.leiden(adata_pred, resolution=res, key_added="leiden", random_state=opt.seed)
        obs_df = adata_pred.obs.dropna()

        davies_bouldin = sklearn.metrics.davies_bouldin_score(adata.obsm["pred"], obs_df["leiden"])
        print("Davies_bouldin: %.2f" % davies_bouldin)

        if davies_bouldin < best_davies:
            if not os.path.exists(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path, "./best.pth"))
            torch.save(model.state_dict(), os.path.join(save_path, "./best.pth"))
            best_adata = adata_pred.copy()
            best_davies = davies_bouldin
            ARI = adjusted_rand_score(obs_df["merge_cell_type"], obs_df["leiden"])
            print("ARI: %.2f" % ARI)

        ARI = adjusted_rand_score(obs_df["leiden"], obs_df["merge_cell_type"])
        print("ARI: %.2f" % ARI)

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print("Select model time required in hours:", hours)

    return best_adata


def train_nano_fov(
    opt,
    adatas,
    hidden_dims=[512, 30],
    n_epochs=1000,
    lr=0.001,
    gradient_clipping=5.0,
    weight_decay=0.0001,
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/nanostring_lung13",
    repeat=1,
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

    # Pre-process
    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)

    hidden_dims = opt.neurons
    hidden_dims_arr = hidden_dims.split(",")
    arr = []
    for i in hidden_dims_arr:
        arr.append(int(i))
    hidden_dims = arr

    # Load VGG-16 instance
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    vgg_extractor = vgg_extractor.to(device)

    datas = []
    gene_dim = 0
    img_dim = 0
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]

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
            img = img_transformed[i].reshape(3, w * 2, h * 2)
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img))   
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                feature = vgg_extractor(img)

            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1))
        features = np.array(features)
        features = np.vstack(features).astype(np.float64)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]

        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)

    loader = DataLoader(datas, batch_size=1, num_workers=0, shuffle=True)
    data = data.to(device)
    img = img.to(device)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), os.path.join(save_path, "init.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train model
    for epoch in tqdm(range(1, n_epochs + 1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()

            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index

            bgene = bgene.float()
            bimg = bimg.float()
            gz, iz, cz, gout, iout, cout, gap = model(bgene.float(), bimg.float(), edge_index, batch.batch)

            # Loss function
            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)

            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            gout_pred = torch.nn.functional.log_softmax(gout, dim=1)
            iout_pred = torch.nn.functional.log_softmax(iout, dim=1)
            cout_pred = torch.nn.functional.log_softmax(cout, dim=1)
            gout_target = torch.nn.functional.softmax(gout, dim=1)
            iout_target = torch.nn.functional.softmax(iout, dim=1)
            cout_target = torch.nn.functional.softmax(cout, dim=1)

            kl_loss1 = kl_loss(gout_pred, cout_target)
            kl_loss2 = kl_loss(cout_pred, iout_target)
            kl_loss3 = kl_loss(iout_pred, gout_target)
            kl_loss4 = kl_loss(cout_pred, gout_target)
            kl_loss5 = kl_loss(iout_pred, cout_target)
            kl_loss6 = kl_loss(gout_pred, iout_target)

            loss = kl_loss1 / 1000 + kl_loss2 / 1000 + kl_loss3 / 1000 + kl_loss4 / 1000 + kl_loss5 / 1000 + kl_loss6 / 1000 + gloss + iloss + closs
            # Backpropagate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        # Save last 10 epoch models for model selection
        if epoch > 190 and epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "final_%d_%d.pth" % (epoch, repeat)),
            )
    torch.save(model.state_dict(), os.path.join(save_path, "final_%d.pth" % (repeat)))
    return adata


def compute_explanations(device, model, cluster, out, bgene, edge_index):
    model.eval()
    vsize = out.size()
    device = device
    grd = torch.zeros(vsize).to(device)
    out.to(device)

    # For each cluster backpropagate to get gradients wrt gene expression input
    grd[:, cluster] = 1
    out.backward(gradient=grd, retain_graph=True)

    final_conv_grads = model.pretrained_model.get_activations_gradient()
    final_conv_acts = model.pretrained_model.get_activations(bgene, edge_index).detach()
    bgene.grad.zero_()
    cell_importance, gene_importance = grad_cam(final_conv_acts, final_conv_grads)
    return cell_importance, gene_importance


def grad_cam(final_conv_acts, final_conv_grads):
    cell_importance = []
    gene_importance = []
    gene_importance = F.relu(final_conv_grads * final_conv_acts)
    cell_importance = F.relu((final_conv_grads * final_conv_acts).sum(dim=1, keepdims=True))
    return cell_importance, gene_importance


def get_explanations(
    opt,
    adatas,
    linmodel,
    model_name=None,
    hidden_dims=[512, 30],
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/nanostring_train_lung13/all//",
):
    start_time = time.time()

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
    datas = []
    gene_dim = 0
    img_dim = 0

    w, h = opt.img_size.split(",")
    w = int(w)
    h = int(h)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # Get VGG model instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    vgg_extractor = vgg_extractor.to(device)

    for adata in adatas:
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
            img = img_transformed[i].reshape(3, w * 2, h * 2)
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img))   
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                feature = vgg_extractor(img)

            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1))

        features = np.array(features)
        features = np.vstack(features).astype(np.float64)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]

        hidden_dims = opt.neurons
        hidden_dims_arr = hidden_dims.split(",")
        arr = []
        for i in hidden_dims_arr:
            arr.append(int(i))
        hidden_dims = arr

        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)

    adata = anndata.concat(adatas)
    loader = DataLoader(datas, batch_size=1, num_workers=0, shuffle=False)

    # Get hybrid graph autoencoder model instance
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    # Load saved model weights
    if model_name is not None:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
    else:
        model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain), map_location=device))

    model.eval()

    # Get explainable model instance and load weights
    clustermodel = xSiGraModel(
        hidden_dims=[gene_dim, img_dim] + hidden_dims,
        pretrained_model=model,
        num_clusters=opt.ncluster,
    ).to(device)

    # Load the linear layer weights
    clustermodel.out1.weight.data = linmodel.out1.weight.data.to(device)
    clustermodel.out2.weight.data = linmodel.out2.weight.data.to(device)
    clustermodel.out1.bias.data = linmodel.out1.bias.data.to(device)
    clustermodel.out2.bias.data = linmodel.out2.bias.data.to(device)

    # Freeze model
    for name, param in clustermodel.named_parameters():
        param.requires_grad = False

    clustermodel.out1.weight.requires_grad = False
    clustermodel.out1.bias.requires_grad = False
    clustermodel.out2.weight.requires_grad = False
    clustermodel.out2.bias.requires_grad = False

    cluster_weights_cell = {}
    cluster_weights_gene = {}

    for i in range(0, opt.ncluster):
        cluster_weights_cell[i] = None

    for i in range(0, opt.ncluster):
        cluster_weights_gene[i] = None

    # Stores the cluster label for each cell
    outlabel = []
    start_time = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        edge_index = batch.edge_index

        bgene = bgene.float()
        bgene.requires_grad = True
        bimg = bimg.float()
        clusterlabel = clustermodel(bgene.float(), bimg.float(), edge_index, batch.batch)

        for label in clusterlabel:
            _, pred = torch.max(label, dim=0)
            outlabel.append(pred.item())

        # For each cluster get node and gene importance scores
        for k in range(0, opt.ncluster):
            cell_explain, gene_explain = compute_explanations(device, clustermodel, k, clusterlabel, bgene.float(), edge_index)
            if cluster_weights_cell[k] is None:
                cluster_weights_cell[k] = cell_explain
            else:
                cluster_weights_cell[k] = torch.cat((cluster_weights_cell[k], cell_explain))

            if cluster_weights_gene[k] is None:
                cluster_weights_gene[k] = gene_explain
            else:
                cluster_weights_gene[k] = torch.cat((cluster_weights_gene[k], gene_explain))

    labels = adata.obs["merge_cell_type"]

    # Match clusters to cell types
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
    match = _hungarian_match(leiden.astype(np.int8), gt.astype(np.int8), opt.ncluster, opt.ncluster)

    # Maps cluster number to cell type using humgarian matching
    dict_cluster_map_cell = {}
    for mapping in match:
        dict_cluster_map_cell[mapping[0]] = le.classes_[mapping[1]]

    # Cell importance score
    for i in range(0, opt.ncluster):
        adata.obs["cmap" + str(i)] = np.array(cluster_weights_cell[i].cpu())

    # Gene importance score
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

        # Sort with highest variance
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

        if not os.path.exists(os.path.join("../cluster_results_gradcam_gt1/" + opt.dataset)):
            os.makedirs(os.path.join("../cluster_results_gradcam_gt1/" + opt.dataset))
        cluster_metadata_df.to_csv("../cluster_results_gradcam_gt1/" + opt.dataset + "/cluster" + str(k) + "_" + dict_cluster_map_cell[k] + ".csv")

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print("Downstream task time in hours:", hours)
    return


def test_img(
    adata,
    hidden_dims=[512, 30],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    random_seed=0,
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

    # Pre-process
    adata.X = sp.csr_matrix(adata.X)
    if "highly_variable" in adata.var.columns:
        adata_Vars = adata[:, adata.var["highly_variable"]]
    else:
        adata_Vars = adata
    data, img = Transfer_img_Data(adata_Vars)

    # Get VGG model instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_extractor = vgg_extractor.to(device)

    data, img = Transfer_img_Data(adata_Vars)
    gene_dim = data.x.shape[1]

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
    img_transformed = img.x.numpy()
    w = 25
    h = 25
    for i in range(img_transformed.shape[0]):
        img = img_transformed[i].reshape(3, w * 2, h * 2)
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = transform(np.uint8(img))   
        img = img.unsqueeze(0)
        img = img.to(device)
        # We only extract features, so we don't need gradient
        with torch.no_grad():
            # Extract the feature from the image
            feature = vgg_extractor(img)

        x = feature
        features.append(x.cpu().detach().numpy().reshape(-1))

    features = np.array(features)
    features = np.vstack(features).astype(np.float64)

    img.x = torch.FloatTensor(features)
    img_dim = img.x.shape[1]
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)

    data.x = torch.cat([data.x, img.x], dim=1)
    datas = []
    datas.append(data)
    loader = DataLoader(datas, batch_size=1, num_workers=0, shuffle=False)

    best_adata = None
    best_dav = float("inf")

    # Get best model among the last 10 saved models
    for i in range(691, 701):
        model.load_state_dict(torch.load(os.path.join(save_path, "final_%d_%d.pth" % (i, 0))))

        for i, batch in enumerate(loader):
            batch = batch.to(device)

            data = batch.x[:, :gene_dim]
            img = batch.x[:, gene_dim:]

            data = data.to(device)
            img = img.to(device)
            model.eval()

            # Get latent space using model
            gz, iz, cz, gout, iout, cout, hidden = model(data, img, batch.edge_index, batch.batch)
            adata_Vars.obsm["pred"] = cz.clone().detach().cpu().numpy()

            indexes = pd.isnull(adata_Vars.obs).any(1).to_numpy().nonzero()
            obs_df = adata_Vars.obs.dropna()
            adata_Vars.obsm["pred"] = cz.to("cpu").detach().numpy().astype(np.float64)
            x = adata_Vars.obsm["pred"].copy()
            x = np.delete(x, indexes, axis=0)
            davies_bouldin = sklearn.metrics.davies_bouldin_score(x, obs_df["Ground Truth"])
            if davies_bouldin < best_dav:
                best_dav = davies_bouldin
                best_adata = adata_Vars.copy()
    return best_adata


def train_img(
    adata,
    hidden_dims=[512, 30],
    n_epochs=1000,
    lr=0.001,
    gradient_clipping=5.0,
    weight_decay=0.0001,
    verbose=True,
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    repeat=1,
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

    adata.X = sp.csr_matrix(adata.X)

    # Pre-process
    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var["highly_variable"]]
    else:
        adata = adata

    if verbose:
        print("Size of Input: ", adata.shape)
    if "Spatial_Net" not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # Get model instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    vgg_extractor = vgg_extractor.to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, img = Transfer_img_Data(adata)
    gene_dim = data.x.shape[1]

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
    img_transformed = img.x.numpy()
    w = 25
    h = 25
    for i in range(img_transformed.shape[0]):
        img = img_transformed[i].reshape(3, w * 2, h * 2)
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = transform(np.uint8(img))   
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            # Extract features
            feature = vgg_extractor(img)

        x = feature
        features.append(x.cpu().detach().numpy().reshape(-1))

    features = np.array(features)
    features = np.vstack(features).astype(np.float64)

    img.x = torch.FloatTensor(features)
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)
    data.x = torch.cat([data.x, img.x], dim=1)
    datas = []
    datas.append(data)

    # Data loader
    loader = DataLoader(datas, batch_size=1, num_workers=0, shuffle=True)

    torch.save(model.state_dict(), os.path.join(save_path, "init.pth"))

    data = data.to(device)
    img = img.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    seed = random_seed

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Train model
    for epoch in tqdm(range(1, n_epochs + 1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            data = batch.x[:, :gene_dim]
            img = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            data = data.float().to(device)
            img = img.float().to(device)
            gz, iz, cz, gout, iout, cout, hidden = model(data.float(), img.float(), edge_index, batch.batch)

            gloss = F.mse_loss(data, gout)
            iloss = F.mse_loss(data, iout)
            closs = F.mse_loss(data, cout)

            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            gout_pred = torch.nn.functional.log_softmax(gout, dim=1)
            iout_pred = torch.nn.functional.log_softmax(iout, dim=1)
            cout_pred = torch.nn.functional.log_softmax(cout, dim=1)
            gout_target = torch.nn.functional.softmax(gout, dim=1)
            iout_target = torch.nn.functional.softmax(iout, dim=1)
            cout_target = torch.nn.functional.softmax(cout, dim=1)

            kl_loss1 = kl_loss(gout_pred, cout_target)
            kl_loss2 = kl_loss(cout_pred, iout_target)
            kl_loss3 = kl_loss(iout_pred, gout_target)
            kl_loss4 = kl_loss(cout_pred, gout_target)
            kl_loss5 = kl_loss(iout_pred, cout_target)
            kl_loss6 = kl_loss(gout_pred, iout_target)
            loss = kl_loss1 / 1000 + kl_loss2 / 1000 + kl_loss3 / 1000 + kl_loss4 / 1000 + kl_loss5 / 1000 + kl_loss6 / 1000 + gloss + iloss + closs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            # Save model
            if epoch > 690:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_path, "final_%d_%d.pth" % (epoch, repeat)),
                )
                adata.obsm["pred"] = cz.clone().detach().cpu().numpy()

                indexes = pd.isnull(adata.obs).any(1).to_numpy().nonzero()
                obs_df = adata.obs.dropna()
                adata.obsm["pred"] = cz.to("cpu").detach().numpy().astype(np.float64)
                x = adata.obsm["pred"].copy()
                x = np.delete(x, indexes, axis=0)
                davies_bouldin = sklearn.metrics.davies_bouldin_score(x, obs_df["Ground Truth"])
                print("Davies_bouldin: %.2f" % davies_bouldin)

    torch.save(model.state_dict(), os.path.join(save_path, "final_%d.pth" % (repeat)))
    print(os.path.join(save_path, "final_%d.pth" % (repeat)))
    return adata

def train_img2(
    adata,
    hidden_dims=[512, 30],
    n_epochs=500,
    lr=0.001,
    gradient_clipping=5.0,
    weight_decay=0.0001,
    verbose=True,
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    repeat=1,
):
    # Set seed
    seed = random_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get model instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    vgg_extractor = vgg_extractor.to(device)
    
    datas = []
    gene_dim = 0
    img_dim = 0 
    adatas = []
    adatas.append(adata)

    for adata in adatas:
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]

        # Transform the image, so it becomes readable with the model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()                              
        ])
        
        features = [] 
        img_transformed = img.x.numpy()
        
        for i in range(img_transformed.shape[0]):
            img = img_transformed[i]
            img_p_normalized = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img_p_normalized))
            img = img.unsqueeze(0)

            with torch.no_grad():
                # Extract features
                feature = vgg_extractor(img.to(device))
    
            img = feature

            features.append(img.cpu().detach().numpy().reshape(-1)) 
        
        features = np.array(features)
        features = np.vstack(features)
        img = torch.from_numpy(features)
    
        img_dim = features.shape[1]
        
        from tqdm import tqdm
        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)
    
    loader = DataLoader(datas, batch_size=128, num_workers=0, shuffle=True)
    data = data.to(device)
    img = img.to(device)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    
    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    seed = 1234
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model
    for epoch in tqdm(range(1, n_epochs+1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            
            bgene = bgene.float()
            bimg = bimg.float()
            gz,iz,cz, gout,iout,cout,gap = model(bgene.float(), bimg.float(), edge_index, batch.batch)
    
            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)
            
            kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
            gout_pred = torch.nn.functional.log_softmax(gout,dim=1)
            iout_pred = torch.nn.functional.log_softmax(iout,dim=1)
            cout_pred = torch.nn.functional.log_softmax(cout,dim=1)
            gout_target = torch.nn.functional.softmax(gout,dim=1)  
            iout_target = torch.nn.functional.softmax(iout,dim=1)
            cout_target = torch.nn.functional.softmax(cout,dim=1)
            
            kl_loss1 = kl_loss(gout_pred, cout_target)
            kl_loss2 = kl_loss(cout_pred, iout_target)
            kl_loss3 = kl_loss(iout_pred, gout_target)
            kl_loss4 = kl_loss(cout_pred, gout_target)
            kl_loss5 = kl_loss(iout_pred, cout_target)
            kl_loss6 = kl_loss(gout_pred, iout_target)

            loss = kl_loss1/1000 + kl_loss2/1000 + kl_loss3/1000 + kl_loss4/1000 + kl_loss5/1000 + kl_loss6/1000 + gloss + iloss + closs
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        repeat = 1
        # Save last 10 epoch models for model selection
        if epoch > 495 and epoch % 1 == 0:
            if not os.path.exists(os.path.join(save_path)):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))
    repeat = 1
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    print(os.path.join(save_path, "final_%d.pth" % (repeat)))
    return adata

def test_img2(
    opt,
    adata,
    hidden_dims=[512, 30],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    random_seed=0,
):
    
    # Set seed
    seed = random_seed
    import random

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get model instance
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_extractor = FeatureExtractor(model)
    vgg_extractor = vgg_extractor.to(device)

    datas = []
    gene_dim = 0
    img_dim = 0 
    adatas = []
    adatas.append(adata)
    for adata in adatas:

        data, img = Transfer_img_Data(adata)

        gene_dim = data.x.shape[1]

        # Transform the image, so it becomes readable with the model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()                              
        ])
        
        features = [] 
        img_transformed = img.x.numpy()

        for i in range(img_transformed.shape[0]):
            img = img_transformed[i]
            img_p_normalized = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img_p_normalized))
            img = img.unsqueeze(0)

            with torch.no_grad():
                feature = vgg_extractor(img.to(device))
    
            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1)) 
        
        features = np.array(features)
        features = np.vstack(features)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]
        
        from tqdm import tqdm
        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)
        
    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=32, num_workers=0, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    best_davies = float('inf')
    best_adata = None

    # Get best model among the last 10 saved models
    for k in range(496,500):
        opt.pretrain = "final_"+str(k)+"_1.pth" 
        model_name=None
        if model_name is not None:
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        else:
            print(os.path.join(save_path, opt.pretrain))
            model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain),map_location=torch.device(opt.device)))

        seed = 1234
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        hidden_matrix = None
        gene_matrix = None
        img_matrix = None
        couts = None

        for i,batch in enumerate(loader):
            batch = batch.to(device)
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]

            bgene = bgene.float()
            bimg = bimg.float()
            bgene.requires_grad = True
            edge_index = batch.edge_index

            # Get latent space using model
            gz,iz,cz, gout,iout,cout,gap = model(bgene, bimg, edge_index, batch.batch)
            
            if hidden_matrix is None:
                hidden_matrix = cz.detach().cpu()
                gene_matrix = gz.detach().cpu()
                couts = cout.detach().cpu()
                img_matrix = iz.detach().cpu()
            else:
                hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
                gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
                img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
                couts = torch.cat([couts, cout.detach().cpu()], dim=0)

        hidden_matrix = hidden_matrix.numpy()
        gene_matrix = gene_matrix.numpy()
        img_matrix = img_matrix.numpy()
        adata.obsm['pred'] = hidden_matrix
        adata.obsm['gene_pred'] = gene_matrix
        adata.obsm['img_pred'] = img_matrix
        couts = couts.numpy().astype(np.float64)
        couts[couts < 0] = 0
        adata.layers['recon'] = couts
        adata_pred = adata
        
        # Perform clustering
        sc.pp.neighbors(adata_pred, opt.ncluster, use_rep='pred')

        def res_search(adata_pred, ncluster, seed, iter=200):
            start = 0; end = 3
            i = 0
            while(start < end):
                if i >= iter: return res
                i += 1
                res = (start + end) / 2
                print(res)
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
                count = len(set(adata_pred.obs['leiden']))
                if count == ncluster:
                    print('find', res)
                    return res
                if count > ncluster:
                    end = res
                else:
                    start = res
            raise NotImplementedError()
        
        res = res_search(adata_pred, opt.ncluster, opt.seed)
        
        sc.tl.leiden(adata_pred, resolution=res, key_added='leiden', random_state=opt.seed)
        obs_df = adata_pred.obs.dropna()
        
        import sklearn
        davies_bouldin = sklearn.metrics.davies_bouldin_score(adata.obsm['pred'], obs_df["leiden"])
        print('Davies_bouldin: %.2f'%davies_bouldin)
        
        if davies_bouldin <= best_davies :

            best_adata = adata_pred.copy()
            best_davies  = davies_bouldin

        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        raw_preds = adata_pred.obs["leiden"]
        silhouette = silhouette_score(adata.obsm["pred"], raw_preds)
        print('Silhouette Score: %.2f' % silhouette)
        
        davies_bouldin = davies_bouldin_score(adata.obsm["pred"], raw_preds)
        print('Davies-Bouldin Score: %.2f' % davies_bouldin)

        calinski = calinski_harabasz_score(adata.obsm["pred"], raw_preds)
        print('Calinski Score: %.2f' % calinski)
    return best_adata

def train_adata(
    adata,
    hidden_dims=[512, 30],
    n_epochs=500,
    lr=0.001,
    gradient_clipping=5.0,
    weight_decay=0.0001,
    verbose=True,
    random_seed=0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    repeat=1,
    feature_extractor_model='vgg'
):
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if feature_extractor_model=='vgg':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif feature_extractor_model=='resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    feature_extractor = FeatureExtractor(model,feature_extractor_model)
    feature_extractor = feature_extractor.to(device)
    
    datas = []
    gene_dim = 0
    img_dim = 0 
    adatas = []
    adatas.append(adata)
    for adata in adatas:
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]

        # Transform the image, so it becomes readable with the model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()                              
        ])
        
        features = [] 
        img_transformed = img.x.numpy()
        
        for i in range(img_transformed.shape[0]):
            img = img_transformed[i]
            img_p_normalized = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img_p_normalized))
            img = img.unsqueeze(0)

            with torch.no_grad():
                # Extract features
                feature = feature_extractor(img.to(device))
    
            img = feature
        
            features.append(img.cpu().detach().numpy().reshape(-1)) 
        
        features = np.array(features)
        features = np.vstack(features)
        img = torch.from_numpy(features)
    
        img_dim = features.shape[1]
        
        from tqdm import tqdm
        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)
    
    loader = DataLoader(datas, batch_size=128, num_workers=0, shuffle=True)
    data = data.to(device)
    img = img.to(device)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    
    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    seed = 1234
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train
    for epoch in tqdm(range(1, n_epochs+1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            
            bgene = bgene.float()
            bimg = bimg.float()
            gz,iz,cz, gout,iout,cout,gap = model(bgene.float(), bimg.float(), edge_index, batch.batch)
    
            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)
            
            kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
            gout_pred = torch.nn.functional.log_softmax(gout,dim=1)
            iout_pred = torch.nn.functional.log_softmax(iout,dim=1)
            cout_pred = torch.nn.functional.log_softmax(cout,dim=1)
            gout_target = torch.nn.functional.softmax(gout,dim=1)  
            iout_target = torch.nn.functional.softmax(iout,dim=1)
            cout_target = torch.nn.functional.softmax(cout,dim=1)
            
            kl_loss1 = kl_loss(gout_pred, cout_target)
            kl_loss2 = kl_loss(cout_pred, iout_target)
            kl_loss3 = kl_loss(iout_pred, gout_target)
            kl_loss4 = kl_loss(cout_pred, gout_target)
            kl_loss5 = kl_loss(iout_pred, cout_target)
            kl_loss6 = kl_loss(gout_pred, iout_target)

            loss = kl_loss1/1000 + kl_loss2/1000 + kl_loss3/1000 + kl_loss4/1000 + kl_loss5/1000 + kl_loss6/1000 + gloss + iloss + closs
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        repeat = 1

        # Save last 10 epoch models for model selection
        if epoch > 495 and epoch % 1 == 0:
            if not os.path.exists(os.path.join(save_path)):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))
    repeat = 1
    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    print(os.path.join(save_path, "final_%d.pth" % (repeat)))
    return adata

def test_adata(
    opt,
    adata,
    hidden_dims=[512, 30],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path="../checkpoint/trans_gene/",
    random_seed=0,
    feature_extractor_model = "vgg"
):
    # Set seed
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if feature_extractor_model=='vgg':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif feature_extractor_model=='resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Get model instance
    feature_extractor = FeatureExtractor(model, feature_extractor_model)
    feature_extractor = feature_extractor.to(device)

    datas = []
    gene_dim = 0
    img_dim = 0 
    adatas = []
    adatas.append(adata)
    for adata in adatas:

        data, img = Transfer_img_Data(adata)

        gene_dim = data.x.shape[1]

        # Transform the image, so it becomes readable with the model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()                              
        ])
        
        features = [] 
        img_transformed = img.x.numpy()

        for i in range(img_transformed.shape[0]):
            img = img_transformed[i]
            img_p_normalized = (img - img.min()) / (img.max() - img.min()) * 255
            img = transform(np.uint8(img_p_normalized))
            img = img.unsqueeze(0)

            with torch.no_grad():
                # Extract the feature from the image
                feature = feature_extractor(img.to(device))
    
            img = feature
            features.append(img.cpu().detach().numpy().reshape(-1)) 
        
        features = np.array(features)
        features = np.vstack(features)
        img = torch.from_numpy(features)

        img_dim = features.shape[1]
        
        from tqdm import tqdm
        data.x = torch.cat([data.x, img], dim=1)
        datas.append(data)
        
    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=32, num_workers=0, shuffle=False)
    
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    best_davies = float('inf')
    best_adata = None

    # Get best model among the last 10 saved models
    for k in range(496,500):
        opt.pretrain = "final_"+str(k)+"_1.pth" 
        model_name=None
        if model_name is not None:
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        else:
            print(os.path.join(save_path, opt.pretrain))
            model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain),map_location=torch.device(opt.device)))

        seed = 1234
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        hidden_matrix = None
        gene_matrix = None
        img_matrix = None
        couts = None

        for i,batch in enumerate(loader):
            batch = batch.to(device)
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]

            bgene = bgene.float()
            bimg = bimg.float()
            bgene.requires_grad = True
            edge_index = batch.edge_index
            
            # Get latent space using model
            gz,iz,cz, gout,iout,cout,gap = model(bgene, bimg, edge_index, batch.batch)
            
            if hidden_matrix is None:
                hidden_matrix = cz.detach().cpu()
                gene_matrix = gz.detach().cpu()
                couts = cout.detach().cpu()
                img_matrix = iz.detach().cpu()
            else:
                hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
                gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
                img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
                couts = torch.cat([couts, cout.detach().cpu()], dim=0)

        hidden_matrix = hidden_matrix.numpy()
        gene_matrix = gene_matrix.numpy()
        img_matrix = img_matrix.numpy()
        adata.obsm['pred'] = hidden_matrix
        adata.obsm['gene_pred'] = gene_matrix
        adata.obsm['img_pred'] = img_matrix
        couts = couts.numpy().astype(np.float32)
        couts[couts < 0] = 0
        adata.layers['recon'] = couts
        adata_pred = adata
        
        # Perform clustering
        sc.pp.neighbors(adata_pred, opt.ncluster, use_rep='pred')

        def res_search(adata_pred, ncluster, seed, iter=200):
            start = 0; end = 3
            i = 0
            while(start < end):
                if i >= iter: return res
                i += 1
                res = (start + end) / 2
                print(res)
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
                count = len(set(adata_pred.obs['leiden']))
                if count == ncluster:
                    print('find', res)
                    return res
                if count > ncluster:
                    end = res
                else:
                    start = res
            raise NotImplementedError()
        
        res = res_search(adata_pred, opt.ncluster, opt.seed)
        
        sc.tl.leiden(adata_pred, resolution=res, key_added='leiden', random_state=opt.seed)
        obs_df = adata_pred.obs.dropna()
        
        import sklearn
        davies_bouldin = sklearn.metrics.davies_bouldin_score(adata.obsm['pred'], obs_df["leiden"])
        print('Davies_bouldin: %.2f'%davies_bouldin)
        
        if davies_bouldin <= best_davies :

            best_adata = adata_pred.copy()
            best_davies  = davies_bouldin

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

        raw_preds = adata_pred.obs["leiden"]
        silhouette = silhouette_score(adata.obsm["pred"], raw_preds)
        print('Silhouette Score: %.2f' % silhouette)
        
        davies_bouldin = davies_bouldin_score(adata.obsm["pred"], raw_preds)
        print('Davies-Bouldin Score: %.2f' % davies_bouldin)

        calinski = calinski_harabasz_score(adata.obsm["pred"], raw_preds)
        print('Calinski Score: %.2f' % calinski)
    return best_adata