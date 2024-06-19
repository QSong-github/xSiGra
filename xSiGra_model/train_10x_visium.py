import os

import numpy as np
import cv2
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models
import scanpy as sc

from utils import Cal_Spatial_Net, Stats_Spatial_Net
from train_transformer import train_img2, test_img2

cudnn.deterministic = True
cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore')

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

def train(opt):
    root = opt.root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Read data
    adata = sc.read_visium(root, load_images=True, count_file=opt.h5ad_file)
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
    best_adata = train_img2(adata, hidden_dims=[512, 30], n_epochs=500, lr=0.001, gradient_clipping=5.0, weight_decay=0.0001, verbose=True, random_seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=opt.save_path, repeat=1)

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
    adata = sc.read_visium(root, load_images=True, count_file=opt.h5ad_file)
    adata.var_names_make_unique()
    adata.X = adata.X.A

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

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    highly_variable_genes = adata.var['highly_variable']
    adata = adata[:, highly_variable_genes]

    # Build graph
    Cal_Spatial_Net(adata, rad_cutoff=opt.rad_cutoff)
    Stats_Spatial_Net(adata)
    
    # Test
    best_adata = test_img2(opt, adata,hidden_dims=[512, 30],device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),save_path=opt.save_path,random_seed=0)    

    end_time = time.time()
    delta = end_time - start_time
    sec = delta
    hours = sec / (60 * 60)
    print('Select model time required in hours:', hours)


def train_10x_visium(opt):
    opt.cluster_method = "mclust"
    if opt.test_only:
        infer(opt)
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        ARI = train(opt)

    return 0