# Follow https://github.com/ys-zong/conST/blob/main/conST_cluster.ipynb
import sys
sys.path.append('../../SiGra_test/SiGra_newmodel/conST')
sys.path.append('../../SiGra_test/SiGra_newmodel/conST/src')

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
import matplotlib.pyplot as plt
import scanpy as sc
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
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
parser.add_argument("--path", type=str, default="../dataset/breast_invasive_carcinoma/")
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--beta', type=float, default=100, help='beta value for l2c')
parser.add_argument('--cont_l2l', type=float, default=0.3, help='Weight of local contrastive learning loss.')
parser.add_argument('--cont_l2c', type=float, default= 0.1, help='Weight of context contrastive learning loss.')
parser.add_argument('--cont_l2g', type=float, default= 0.1, help='Weight of global contrastive learning loss.')

parser.add_argument('--edge_drop_p1', type=float, default=0.1, help='drop rate of adjacent matrix of the first view')
parser.add_argument('--edge_drop_p2', type=float, default=0.1, help='drop rate of adjacent matrix of the second view')
parser.add_argument('--node_drop_p1', type=float, default=0.2, help='drop rate of node features of the first view')
parser.add_argument('--node_drop_p2', type=float, default=0.3, help='drop rate of node features of the second view')
parser.add_argument('--save_path', type=str, default='./HBC_conST')

# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=300, help='Eval graph kN tol.') 

params =  parser.parse_args()

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params.device = device

root = params.path
adata = sc.read_visium(root, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.X = adata.X.A

image = cv2.imread(os.path.join(root, "Visium_FFPE_Human_Breast_Cancer_image.tif"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

label = np.zeros(adata.shape[0], dtype=int)

patches = []
img_size = 112
adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
for x, y in adata.obsm['spatial']:
    x, y = int(x), int(y)
    patches.append(image[y - img_size:y + img_size, x - img_size:x + img_size])
patches = np.array(patches)
image = patches
adata.obs["label"] = label

adata_X, i_adata = adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
label = adata.obs['label']

n_clusters = 10

graph_dict = graph_construction(i_adata.obsm['spatial'], i_adata.shape[0], params)
np.save('./patch_HBC.npy', np.array(patches))

# Run MAE to extract features - https://github.com/ys-zong/conST/tree/main/MAE-pytorch

np.save('./adatax_HBC.npy', adata_X)

with open('./graphdict_HBC.pkl', 'wb') as f:
    pickle.dump(graph_dict, f, protocol=4)

params.save_path = mk_dir(f'{params.save_path}')

adata_X = np.load('./adatax_HBC.npy')

with open('graphdict_HBC.pkl', 'rb') as f:
    graph_dict = pickle.load(f)
params.cell_num = adata.shape[0]

if params.use_img:
    img_transformed = np.load(OUTPUT_DIR)
    print(img_transformed.shape)
    img_transformed = (img_transformed - img_transformed.mean()) / img_transformed.std() * adata_X.std() + adata_X.mean()
    conST_net = conST_training(adata_X, adata, graph_dict, params, n_clusters, img_transformed)
else:
    conST_net = conST_training(adata_X, adata, graph_dict, params, n_clusters)
if params.use_pretrained:
    conST_net.load_model('conST_151673.pth')
else:
    conST_net.pretraining()
    conST_net.major_training()

conST_embedding, adata = conST_net.get_embedding()

# Clustering
adata.obsm["pred"] = conST_embedding
sc.pp.neighbors(adata, use_rep='pred')
cluster_key = "conST_leiden"

def res_search(adata_pred, ncluster, seed, iter=200):
    start = 0; end = 3
    i = 0
    while(start < end):
        if i >= iter: return res
        i += 1
        res = (start + end) / 2
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
        count = len(set(adata_pred.obs['leiden']))
        if count == ncluster:
            return res
        if count > ncluster:
            end = res
        else:
            start = res
    raise NotImplementedError()

res = res_search(adata, 8, 1234)
sc.tl.leiden(adata, key_added=cluster_key, random_state=1234,resolution=res)

obs_df = adata.obs.dropna()
raw_preds = obs_df["conST_leiden"]

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(adata.obsm["pred"], raw_preds)
print('Silhouette Score: %.2f' % silhouette)

davies_bouldin = davies_bouldin_score(adata.obsm["pred"], raw_preds)
print('Davies-Bouldin Score: %.2f' % davies_bouldin)

calinski = calinski_harabasz_score(adata.obsm["pred"], raw_preds)
print('Calinski Score: %.2f' % calinski)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (3, 3)
sc.settings.figdir = params.save_path
adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)
ax=sc.pl.spatial(adata, color=['conST_leiden'], title=['conST'], show=False)
plt.savefig(os.path.join(params.save_path, 'HBC_Const_spatial.pdf'), bbox_inches='tight')