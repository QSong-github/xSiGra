import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from statsmodels.stats.multitest import multipletests

# Load cell-cell interaction scores
fov1 = pd.read_csv("./enrichment_results/fov12345.csv")
fov2 = pd.read_csv("./enrichment_results/fov678910.csv")
fov3 = pd.read_csv("./enrichment_results/fov1112131415.csv")
fov4 = pd.read_csv("./enrichment_results/fov1617181920.csv")

cci_results = pd.concat([fov1, fov2, fov3, fov4])

cci_results["ligand-receptor"] = cci_results["ligands"] + "_" + cci_results["celltype1"] + "_" + cci_results["receptor"] + "_" + cci_results["celltype2"]

# Compute results using importance score based Cell-Cell Interaction score
test = cci_results
score = test.groupby("ligand-receptor").mean().reset_index()
lr1 = score["ligand-receptor"].to_list()
score1 = score["score_1"].to_list()
score2 = score["score_2"].to_list()
scores = score1 + score2
lr = lr1 + lr1
assert len(scores) == len(lr), "Length mismatch between scores and lr_extended"

list1, list2 = (list(t) for t in zip(*sorted(zip(scores, lr))))

# Compute zscore and pvalue
zscore = stats.zscore(scores)
pvalue = stats.norm.sf(abs(zscore))

# Compute FDR
fdr = multitest.fdrcorrection(pvalue)[1]
fdr = multipletests(pvalue, method="bonferroni")[1]
fdr_threshold = 0.05
fdr_columns = []


celltype1 = []
celltype2 = []
ligand = []
receptor = []
pvalues = []
zscores = []
fdrs = []

# Select Ligand celltype Receptor celltype pairs with FDR < 0.05
for i in range(len(fdr)):
    if fdr[i] <= fdr_threshold:
        ligand.append(lr[i].split("_")[0].split(" ")[0])
        celltype1.append(lr[i].split("_")[1].split(" ")[0])
        receptor.append(lr[i].split("_")[2].split(" ")[0])
        celltype2.append(lr[i].split("_")[3].split(" ")[0])
        pvalues.append(pvalue[i])
        zscores.append(zscore[i])
        fdrs.append(fdr[i])

df = pd.DataFrame(
    {
        "ligand": ligand,
        "celltype1": celltype1,
        "receptor": receptor,
        "celltype2": celltype2,
        "pvalue": pvalues,
        "zscore": zscores,
        "fdr": fdrs,
    }
)

# Save to csv
df = df.sort_values(by=["zscore"], ascending=False)
df.to_csv("./enrichment_results/imp_score_sig_pairs.csv")

# Compute results using raw expression based Cell-Cell Interaction score
test = cci_results
score = test.groupby("ligand-receptor").mean().reset_index()
lr1 = score["ligand-receptor"].to_list()

score1 = score["raw_exp_score_1"].to_list()
score2 = score["raw_exp_score_2"].to_list()
scores = score1 + score2
lr = lr1 + lr1

assert len(scores) == len(lr), "Length mismatch between scores and lr_extended"

list1, list2 = (list(t) for t in zip(*sorted(zip(scores, lr))))

# Compute zscore and pvalue
zscore = stats.zscore(scores)
pvalue = stats.norm.sf(abs(zscore))

fdr = multitest.fdrcorrection(pvalue)[1]
fdr = multipletests(pvalue, method="bonferroni")[1]
fdr_threshold = 0.05
fdr_columns = []

celltype1 = []
celltype2 = []
ligand = []
receptor = []
pvalues = []
zscores = []
fdrs = []

# Select Ligand celltype Receptor celltype pairs with FDR < 0.05
for i in range(len(fdr)):
    if fdr[i] <= fdr_threshold:
        ligand.append(lr[i].split("_")[0].split(" ")[0])
        celltype1.append(lr[i].split("_")[1].split(" ")[0])
        receptor.append(lr[i].split("_")[2].split(" ")[0])
        celltype2.append(lr[i].split("_")[3].split(" ")[0])
        pvalues.append(pvalue[i])
        zscores.append(zscore[i])
        fdrs.append(fdr[i])

df = pd.DataFrame(
    {
        "ligand": ligand,
        "celltype1": celltype1,
        "receptor": receptor,
        "celltype2": celltype2,
        "pvalue": pvalues,
        "zscore": zscores,
        "fdr": fdrs,
    }
)

# Save results to csv
df = df.sort_values(by=["zscore"], ascending=False)
df.to_csv("./enrichment_results/raw_exp_sig_pairs.csv")
