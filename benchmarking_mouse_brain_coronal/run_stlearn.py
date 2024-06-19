import argparse
import os
import cv2
import pandas as pd
from pathlib import Path
import scanpy as sc
import stlearn as st
import matplotlib.pyplot as plt

def run_stlearn(adata, image, BASE_PATH, save_path, ncluster):
    TILE_PATH = Path("/tmp/{}_tiles".format(""))
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("%s" % (save_path))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Read data
    data = adata

    # Read image
    img = image

    arr = data.obsm["spatial"]

    coors = data.obsm['spatial']
    data.obs['px'] = coors[:, 0]
    data.obs['py'] = coors[:, 1]
    
    # Create a DataFrame from arr
    df = pd.DataFrame(arr, index=adata.obs.index, columns=["cx", "cy"])
    
    # Extract cx and cy from the DataFrame and assign them to adata.obs
    data.uns['spatial'] = {id: {'images': {'hires': img/255.0}, 'use_quality': 'hires'}}
    data.obs['imagerow'] = data.obs['px']
    data.obs['imagecol'] = data.obs['py']

    n_cluster = ncluster

    # Pre-process
    st.pp.filter_genes(data, min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)
    st.em.run_pca(data, n_comps=150)
    st.pp.tiling(data, TILE_PATH)
    st.pp.extract_feature(data)

    # Run SME learning
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm["raw_SME_normalized"]
    st.pp.scale(data_)

    # Run PCA
    st.em.run_pca(data_, n_comps=30)

    # Kmeans clustering
    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")
    df = data_.obs.dropna()
    data_ = data_[df.index, :]
    
    # Compute ARI
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    silhouette = silhouette_score(data_.obsm["X_pca"], df["X_pca_kmeans"])
    print('Silhouette Score: %.2f' % silhouette)
    
    davies_bouldin = davies_bouldin_score(data_.obsm["X_pca"], df["X_pca_kmeans"])
    print('Davies-Bouldin Score: %.2f' % davies_bouldin)

    calinski = calinski_harabasz_score(data_.obsm["X_pca"], df['X_pca_kmeans'])
    print('Calinski Score: %.2f' % calinski)

    return data_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../dataset/breast_invasive_carcinoma/")
    parser.add_argument("--save_path", type=str, default="./HBC_stLearn/")
    parser.add_argument("--ncluster", type=int, default=8)
    opt = parser.parse_args(args=[])

    
    root = opt.path
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adata = sc.read_visium(root, load_images=True, count_file="Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    adata1 = adata.copy()
    adata.var_names_make_unique()
    adata.X = adata.X.A

    import cv2
    image = cv2.imread(os.path.join(root, "Visium_FFPE_Human_Breast_Cancer_image.tif"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ncluster = opt.ncluster

    # Run stlearn and compute ARI
    adata = run_stlearn(adata, image, "BASE_PATH", opt.save_path, ncluster)

    adata1.obs["X_pca_kmeans"] = adata.obs["X_pca_kmeans"]
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax=sc.pl.spatial(adata1, color=['X_pca_kmeans'], title=['stLearn'], show=False)
    plt.savefig(os.path.join(save_path, 'stLearn_spatial.pdf'), bbox_inches='tight')