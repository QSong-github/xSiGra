# Follow https://github.com/ys-zong/conST/blob/main/conST_cluster.ipynb
import torch
import argparse
import random
import numpy as np
import pandas as pd
import pickle
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
from src.training import conST_training

import anndata
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import os
import warnings
warnings.filterwarnings('ignore')

import os
import cv2

import torchvision.transforms as transforms
from sklearn.metrics.cluster import adjusted_rand_score

# Set R path

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--use_img', type=bool, default=True, help='Use histology images.')
parser.add_argument('--img_w', type=float, default=0.1, help='Weight of image features.')
parser.add_argument('--use_pretrained', type=bool, default=False, help='Use pretrained weights.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--beta', type=float, default=100, help='beta value for l2c')
parser.add_argument('--cont_l2l', type=float, default=0.3, help='Weight of local contrastive learning loss.')
parser.add_argument('--cont_l2c', type=float, default= 0.1, help='Weight of context contrastive learning loss.')
parser.add_argument('--cont_l2g', type=float, default= 0.1, help='Weight of global contrastive learning loss.')

parser.add_argument('--edge_drop_p1', type=float, default=0.1, help='drop rate of adjacent matrix of the first view')
parser.add_argument('--edge_drop_p2', type=float, default=0.1, help='drop rate of adjacent matrix of the second view')
parser.add_argument('--node_drop_p1', type=float, default=0.2, help='drop rate of node features of the first view')
parser.add_argument('--node_drop_p2', type=float, default=0.3, help='drop rate of node features of the second view')

# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

parser.add_argument('--root', type=str, default='../dataset/10x/')
parser.add_argument('--id', type=int, default="151676", help='sample id') 
parser.add_argument('--ncluster', type=int, default=7, help='number of clusters') 
params =  parser.parse_args()

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('Using device: ' + device)
params.device = device

from PIL import Image
def gen_patches(adata, root, id):
    img = cv2.imread(os.path.join(root, str(id), "spatial/full_image.tif"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    patchs = []

    w, h = 112,112
    w = int(w)
    h = int(h)
    
    for coor in adata.obsm['spatial']:
        x, y = coor
        
        img_p = img[int(y-h):int(y+h), int(x-w): int(x+w),:]
        patchs.append(img_p)

    return adata, patchs


adata_h5 = sc.read(os.path.join(params.root, str(params.id), "sampledata.h5ad"))
Ann_df = pd.read_csv("%s/%s/annotation.txt" % (params.root, params.id), sep="\t", header=None, index_col=0)
Ann_df.columns = ["Ground Truth"]
adata_h5.obs["Ground Truth"] = Ann_df.loc[adata_h5.obs_names, "Ground Truth"]

adata_X, i_adata = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)

i_adata, patch=  gen_patches(i_adata, params.root, params.id)
np.save('./patch'+str(params.id)+'.npy', np.stack(patch))
graph_dict = graph_construction(i_adata.obsm['spatial'], i_adata.shape[0], params)

# Run MAE to extract features - https://github.com/ys-zong/conST/tree/main/MAE-pytorch

np.save('./adatax_visium'+ str(params.id) +'.npy', adata_X)

with open('./graphdict_visium'+ str(params.id) +'.pkl', 'wb') as f:
    pickle.dump(graph_dict, f, protocol=4)

adata_X = np.load('./adatax_visium'+ str(params.id) +'.npy')

with open('graphdict_visium'+ str(params.id) +'.pkl', 'rb') as f:
    graph_dict = pickle.load(f)
params.cell_num = adata_h5.shape[0]

n_clusters = params.ncluster
if params.use_img:
    img_transformed = np.load(OUTPUT_DIR)
    img_transformed = (img_transformed - img_transformed.mean()) / img_transformed.std() * adata_X.std() + adata_X.mean()
    conST_net = conST_training(adata_X, i_adata, graph_dict, params, n_clusters, img_transformed)
else:
    conST_net = conST_training(adata_X, i_adata, graph_dict, params, n_clusters)
if params.use_pretrained:
    conST_net.load_model('conST_151673.pth')
else:
    conST_net.pretraining()
    conST_net.major_training()

conST_embedding, adata = conST_net.get_embedding()

# clustering
sc.pp.neighbors(adata, n_neighbors=params.eval_graph_n, use_rep='pred')
eval_resolution = res_search_fixed_clus(adata, n_clusters)
cluster_key = "conST_leiden"
sc.tl.leiden(adata, key_added=cluster_key, resolution=eval_resolution)

# plotting
from sklearn.metrics.cluster import adjusted_rand_score
obs_df = adata.obs.dropna()
adata = adata[obs_df.index, :]

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ARI = adjusted_rand_score(obs_df['conST_leiden'], obs_df['Ground Truth'])
print(params.id,'ARI: %.2f'%ARI)

NMI = normalized_mutual_info_score(obs_df['conST_leiden'], obs_df['Ground Truth'])
print('NMI: %.2f' % NMI)