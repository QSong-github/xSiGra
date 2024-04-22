import os
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as sm

# for dataset in ["lung5-rep1","lung5-rep2","lung5-rep3","lung6","lung9-rep1","lung9-rep2","lung12","lung13"]:
for dataset in ["lung13"]:
    # For each cluster
    list_1 = [
        "cluster7_tumors",
        "cluster2_fibroblast",
        "cluster3_lymphocyte",
        "cluster5_myeloid",
        "cluster4_mast",
        "cluster0_endothelial",
        "cluster1_epithelial",
        "cluster6_neutrophil",
    ]
    for cluster in list_1:
        # Read scores file for cluster for gradcam
        gene_explanations = pd.read_csv("../cluster_results_gradcam_gt1/" + dataset + "/" + cluster + ".csv")

        # Use only importance scores for genes
        gene_explanations = gene_explanations.iloc[:, 10:]

        # Use only importance scores for genes
        variance = gene_explanations.var()

        # Get names
        gene_name = variance.index

        # Compute zscore and pvalue
        zscores = stats.zscore(variance)
        pvals = [stats.norm.sf(abs(i)) for i in zscores]

        # FDR
        reject, pvals_adj = sm.fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False)

        # Select genes with fdr < 0.05
        imp_genes = []
        for i in range(len(gene_name)):
            if pvals_adj[i] < 0.05:
                imp_genes.append(gene_name[i])

        # Save to file
        imp_genes = pd.DataFrame({"sig_genes": imp_genes})
        if not os.path.exists("./enrichment_results/"):
            os.makedirs("./enrichment_results/")
        imp_genes.to_csv("enrichment_results/sig_genes_gradcam_" + dataset + "_" + cluster + ".csv")

        # Read scores file for cluster for deconvolution
        gene_explanations = pd.read_csv("../cluster_results_deconvolution_gt1/" + dataset + "/" + cluster + ".csv")

        # Use only importance scores for genes
        gene_explanations = gene_explanations.iloc[:, 10:]

        # Use only importance scores for genes
        variance = gene_explanations.var()

        # Get names
        gene_name = variance.index

        # Compute zscore and pvalue
        zscores = stats.zscore(variance)
        pvals = [stats.norm.sf(abs(i)) for i in zscores]

        # FDR
        reject, pvals_adj = sm.fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False)

        # Select genes with fdr < 0.05
        imp_genes = []
        for i in range(len(gene_name)):
            if pvals_adj[i] < 0.05:
                imp_genes.append(gene_name[i])

        # Save to file
        imp_genes = pd.DataFrame({"sig_genes": imp_genes})
        if not os.path.exists("./enrichment_results/"):
            os.makedirs("./enrichment_results/")
        imp_genes.to_csv("enrichment_results/sig_genes_deconvolution_" + dataset + "_" + cluster + ".csv")

        # Read scores file for cluster guidedbackprop
        gene_explanations = pd.read_csv("../cluster_results_guidedbackprop_gt1/" + dataset + "/" + cluster + ".csv")

        # Use only importance scores for genes
        gene_explanations = gene_explanations.iloc[:, 10:]

        # Use only importance scores for genes
        variance = gene_explanations.var()

        # Get names
        gene_name = variance.index

        # Compute zscore and pvalue
        zscores = stats.zscore(variance)
        pvals = [stats.norm.sf(abs(i)) for i in zscores]

        # FDR
        reject, pvals_adj = sm.fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False)

        # Select genes with fdr < 0.05
        imp_genes = []
        for i in range(len(gene_name)):
            if pvals_adj[i] < 0.05:
                imp_genes.append(gene_name[i])

        # Save to file
        imp_genes = pd.DataFrame({"sig_genes": imp_genes})
        if not os.path.exists("./enrichment_results/"):
            os.makedirs("./enrichment_results/")
        imp_genes.to_csv("enrichment_results/sig_genes_guidedbackprop_" + dataset + "_" + cluster + ".csv")

        # Read scores file for cluster for inputxgradient
        gene_explanations = pd.read_csv("../cluster_results_inputxgradient_gt1/" + dataset + "/" + cluster + ".csv")

        # Use only importance scores for genes
        gene_explanations = gene_explanations.iloc[:, 10:]

        # Use only importance scores for genes
        variance = gene_explanations.var()

        # Get names
        gene_name = variance.index

        # Compute zscore and pvalue
        zscores = stats.zscore(variance)
        pvals = [stats.norm.sf(abs(i)) for i in zscores]

        # FDR
        reject, pvals_adj = sm.fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False)

        # Select genes with FDR < 0.05
        imp_genes = []
        for i in range(len(gene_name)):
            if pvals_adj[i] < 0.05:
                imp_genes.append(gene_name[i])

        # Save to file
        imp_genes = pd.DataFrame({"sig_genes": imp_genes})
        if not os.path.exists("./enrichment_results/"):
            os.makedirs("./enrichment_results/")
        imp_genes.to_csv("enrichment_results/sig_genes_inputx_gradient_" + dataset + "_" + cluster + ".csv")

        # Read scores file for cluster for saliency
        gene_explanations = pd.read_csv("../cluster_results_saliency_gt1/" + dataset + "/" + cluster + ".csv")

        # Use only importance scores for genes
        gene_explanations = gene_explanations.iloc[:, 10:]

        # Use only importance scores for genes
        variance = gene_explanations.var()
        gene_name = variance.index

        zscores = stats.zscore(variance)
        pvals = [stats.norm.sf(abs(i)) for i in zscores]

        # FDR
        reject, pvals_adj = sm.fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False)

        # Select genes with fdr < 0.05
        imp_genes = []
        for i in range(len(gene_name)):
            if pvals_adj[i] < 0.05:
                imp_genes.append(gene_name[i])

        # Save to file
        imp_genes = pd.DataFrame({"sig_genes": imp_genes})

        if not os.path.exists("./enrichment_results/"):
            os.makedirs("./enrichment_results/")
        imp_genes.to_csv("enrichment_results/sig_genes_saliency_" + dataset + "_" + cluster + ".csv")
