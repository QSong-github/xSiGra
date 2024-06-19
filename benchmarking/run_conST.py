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

import cv2
import torchvision.transforms as transforms
from sklearn.metrics.cluster import adjusted_rand_score

# Set R path

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
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
parser.add_argument('--n_clusters', type=int, default=8, help='Eval graph kN tol.') 

parser.add_argument('--num_fov', type=int, default=20, help='fov') 
parser.add_argument('--root', type=str, default='../dataset/nanostring/lung13')
parser.add_argument('--name', type=str, default='lung13')

params =  parser.parse_args()

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)
params.device = device


from PIL import Image
def gen_patches(adata, root, id, img_name):
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    patchs = []

    w, h = 112, 112
    w = int(w)
    h = int(h)
    
    for coor in adata.obsm['spatial']:
        x, y = coor
        
        img_p = img[int(y-h):int(y+h), int(x-w): int(x+w),:]
        
        transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((224,224)),
          transforms.ToTensor()
        ])

        img_p = img_p.reshape(w*2,h*2,3)
    
        img_p = transform(img_p)
        img_p = img_p.permute(1, 2, 0).numpy()

        
        patchs.append(img_p)

    return adata, patchs

img_names =[]
ids = ['fov'+str(i) for i in range(1,int(params.num_fov)+1)]
img_names = ['F00'+str(i) for i in range(1,10)]
img_names = img_names + ['F0'+str(i) for i in range(10,100)]
img_names = img_names + ['F'+str(i) for i in range(100,int(params.num_fov)+1)]

adatas = list()
patches = []


for id, name in zip(ids, img_names):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))

    adata_X, i_adata = adata_preprocess(adata_h5, min_cells=3, pca_n_comps=params.cell_feat_dim)
    i_adata, patch =  gen_patches(i_adata, params.root, id, name)
    
    graph_dict = graph_construction(i_adata.obsm['spatial'], i_adata.shape[0], params)
    np.save(params.name+'.npy', np.array(patch))
    
    # Run MAE to extract features - https://github.com/ys-zong/conST/tree/main/MAE-pytorch
    
    np.save(params.name+'_adatax.npy', adata_X)
    
    with open(params.name+'_graphdict.pkl', 'wb') as f:
        pickle.dump(graph_dict, f, protocol=4)
    np.save(params.name+'_graphdict.npy', graph_dict, allow_pickle = True)

 
    adata_X = np.load(params.name+'_adatax.npy')
    with open(params.name+'_graphdict.pkl', 'rb') as f:
        graph_dict = pickle.load(f)
    params.cell_num = adata_h5.shape[0]
    
    n_clusters = params.n_clusters
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
    embedding.append(conST_embedding)
    adata.obsm["pred"] = conST_embedding
    
    sc.pp.neighbors(adata, use_rep='pred')
    
    increment=0.01
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=1234, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())

        if count_unique_leiden == len(set(adata.obs["merge_cell_type"])):
            break
    eval_resolution = res


    cluster_key = "conST_leiden"
    sc.tl.leiden(adata, random_state=1234, key_added=cluster_key, resolution=eval_resolution)

    from sklearn.metrics.cluster import adjusted_rand_score
    obs_df = adata.obs.dropna()
    
    ARI = adjusted_rand_score(obs_df['conST_leiden'], obs_df['merge_cell_type'])
    run_ConGI'ARI: %.2f'%ARI)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    gt = le.fit_transform(obs_df['merge_cell_type_int'])

    def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
        num_samples = flat_target.shape[0]
        num_k = preds_k
        num_k_tar = target_k
        num_correct = np.zeros((num_k, num_k_tar))
        for c1 in range(num_k):
            for c2 in range(num_k_tar):
                votes = int(((flat_preds==c1)*(flat_target==c2)).sum())
                num_correct[c1, c2] = votes
        match = linear_sum_assignment(num_samples-num_correct)
        match = np.array(list(zip(*match)))
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))
        return res

    from scipy.optimize import linear_sum_assignment    
    
    match = _hungarian_match(adata.obs["conST_leiden"].astype(np.int8), gt.astype(np.int8), params.n_clusters,len(obs_df['merge_cell_type'].unique()))

    ARI = adjusted_rand_score(obs_df['conST_leiden'], obs_df['merge_cell_type'])
    run_ConGI'ARI: %.2f'%ARI)
    
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
    
    obs_df = obs_df.replace({"conST_leiden": dict_mapping})
    adata.obs["conST_leiden"] = obs_df["conST_leiden"]
    adatas.append(adata)
    
final_embedding = np.vstack(embedding)

import anndata
adata = anndata.concat(adatas)
adata.obsm["pred"] = final_embedding

from sklearn.metrics.cluster import adjusted_rand_score
adata.obs["conST_leiden"] = adata.obs["conST_leiden"].astype(int)
obs_df = adata.obs.dropna()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ARI = adjusted_rand_score(obs_df['conST_leiden'], obs_df['merge_cell_type'])
run_ConGIname+'ARI: %.2f'%ARI)

NMI = normalized_mutual_info_score(obs_df['conST_leiden'], obs_df['merge_cell_type'])
run_ConGI'NMI: %.2f' % NMI)
