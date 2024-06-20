import os

import numpy as np
import cv2
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models
import scanpy as sc

from utils import Cal_Spatial_Net, Stats_Spatial_Net
from train_transformer import train_adata, test_adata

cudnn.deterministic = True
cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore')

from torch_geometric.loader import NeighborLoader, NeighborSampler, DataLoader
from transModel import TransImg, ClusteringLayerModel
from utils import Transfer_img_Data

import torch
import torch.backends.cudnn as cudnn

def train(opt):
    root = opt.root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Read data 
    # Update as per data file
    adata = sc.read(os.path.join(root,opt.h5ad_file))
    adata.var_names_make_unique()
    adata.X = adata.X.A

    image = cv2.imread(os.path.join(root,opt.image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    patches = []
    img_size = int(opt.img_size)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
    for x, y in adata.obsm['spatial']:
        x, y = int(x), int(y)
        patches.append(image[y - img_size:y + img_size, x - img_size:x + img_size])
    patches = np.array(patches)
    image = patches
    adata.obsm["imgs"] = patches

    # Update preprocess based on data
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    highly_variable_genes = adata.var['highly_variable']
    adata = adata[:, highly_variable_genes]

    # Build graph
    Cal_Spatial_Net(adata, rad_cutoff=opt.rad_cutoff)
    Stats_Spatial_Net(adata)

    import time
    start_time = time.time()
    import random
    
    # Train
    best_adata = train_adata(adata, hidden_dims=[512, 30], n_epochs=500, lr=0.001, gradient_clipping=5.0, weight_decay=0.0001, verbose=True, random_seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=opt.save_path, repeat=1, feature_extractor_model="vgg")

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print('difference in hours:', hours)

def infer(opt):
    root = opt.root
    save_path = opt.save_path
    
    import time
    start_time = time.time()
    import random
    
    # Read data
    # Update as per the data file
    adata = sc.read(os.path.join(root,opt.h5ad_file))
    adata.var_names_make_unique()
    adata.X = adata.X.A

    # Add labels in adata.obs["labels"] if available

    # Put image in root folder
    image = cv2.imread(os.path.join(root, opt.image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    patches = []
    img_size = int(opt.img_size)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
    for x, y in adata.obsm['spatial']:
        x, y = int(x), int(y)
        patches.append(image[y - img_size:y + img_size, x - img_size:x + img_size])
    patches = np.array(patches)
    image = patches
    
    adata.obsm["imgs"] = patches

    # Update preprocess based on data
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    highly_variable_genes = adata.var['highly_variable']
    adata = adata[:, highly_variable_genes]

    # Build graph
    Cal_Spatial_Net(adata, rad_cutoff=opt.rad_cutoff)
    Stats_Spatial_Net(adata)
    
    # Test
    best_adata = test_adata(opt, adata,hidden_dims=[512, 30],device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),save_path=opt.save_path,random_seed=0, feature_extractor_model="vgg")    

    # Compute ARI and NMI if ground truth available
    if "labels" in best_adata.obs:
        obs_df = best_adata.obs.dropna()
        
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ARI = adjusted_rand_score(obs_df['leiden'], obs_df['labels'])
        print('ARI: %.2f' % ARI)
        
        NMI = normalized_mutual_info_score(obs_df['leiden'], obs_df['labels'])
        print('NMI: %.2f' % NMI)

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print('Select model time required in hours:', hours)


def train_h5ad(opt):
    opt.cluster_method = "mclust"
    if opt.test_only:
        infer(opt)
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        ARI = train(opt)

    return 0