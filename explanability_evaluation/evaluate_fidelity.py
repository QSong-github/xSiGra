import os
import statistics

import pandas as pd

# for dataset in ["lung5-rep1","lung5-rep2","lung5-rep3","lung9-rep1","lung12","lung13","lung6","lung9-rep2"]:
for dataset in ["lung13"]:
    if dataset == "lung9-rep1":
        list_1 = [
            "cluster7_tumors.csv",
            "cluster3_fibroblast.csv",
            "cluster4_lymphocyte.csv",
            "cluster0_Mcell.csv",
            "cluster5_mast.csv",
            "cluster1_endothelial.csv",
            "cluster2_epithelial.csv",
            "cluster6_neutrophil.csv",
        ]
    elif dataset == "lung6" or dataset == "lung9-rep2":
        list_1 = [
            "cluster0_endothelial.csv",
            "cluster1_epithelial.csv",
            "cluster2_fibroblast.csv",
            "cluster3_lymphocyte.csv",
        ]
    else:
        list_1 = [
            "cluster7_tumors.csv",
            "cluster2_fibroblast.csv",
            "cluster3_lymphocyte.csv",
            "cluster5_myeloid.csv",
            "cluster4_mast.csv",
            "cluster0_endothelial.csv",
            "cluster1_epithelial.csv",
            "cluster6_neutrophil.csv",
        ]

    # GradCAM
    gradcam = []
    saliency = []
    inputxgradient = []
    deconvolution = []
    guidedbackprop = []

    pair1 = pd.read_csv("./fidelity_results/fidelity_all_gradcam_" + dataset, index_col=0)
    pair2 = pd.read_csv("./fidelity_results/fidelity_mask_gradcam_" + dataset, index_col=0)
    gradcam.extend(list(pair1["auc"] - pair2["auc"]))

    pair1 = pd.read_csv("./fidelity_results/fidelity_all_saliency_" + dataset, index_col=0)
    pair2 = pd.read_csv("./fidelity_results/fidelity_mask_saliency_" + dataset, index_col=0)
    saliency.extend(list(pair1["auc"] - pair2["auc"]))

    pair1 = pd.read_csv("./fidelity_results/fidelity_all_inputxgradient_" + dataset, index_col=0)
    pair2 = pd.read_csv("./fidelity_results/fidelity_mask_inputxgradient_" + dataset, index_col=0)
    inputxgradient.extend(list(pair1["auc"] - pair2["auc"]))

    pair1 = pd.read_csv("./fidelity_results/fidelity_all_deconvolution_" + dataset, index_col=0)
    pair2 = pd.read_csv("./fidelity_results/fidelity_mask_deconvolution_" + dataset, index_col=0)
    deconvolution.extend(list(pair1["auc"] - pair2["auc"]))

    pair1 = pd.read_csv("./fidelity_results/fidelity_all_guidedbackprop_" + dataset, index_col=0)
    pair2 = pd.read_csv("./fidelity_results/fidelity_mask_guidedbackprop_" + dataset, index_col=0)
    guidedbackprop.extend(list(pair1["auc"] - pair2["auc"]))

    print("GradCAM", statistics.median(gradcam))
    print("Saliency", statistics.median(saliency))
    print("inputxgradient", statistics.median(inputxgradient))
    print("guidedbackprop", statistics.median(guidedbackprop))
    print("deconvolution", statistics.median(deconvolution))

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    fig, ax = plt.subplots()
    df = pd.DataFrame(
        {
            "deconvolution": deconvolution,
            "guidedbackprop": guidedbackprop,
            "saliency": saliency,
            "inputxgradient": inputxgradient,
            "gradcam": gradcam,
        }
    )

    data_df = df.melt(var_name="method", value_name="ARI")

    custom_colors = ["#65b044", "#5b37b7", "#964B00", "#ff9e9e", "#ff1100"]
    sns.boxplot(x="method", y="ARI", data=data_df, ax=ax, palette=custom_colors)

    sns.stripplot(x="method", y="ARI", color="black", data=data_df, ax=ax)

    plt.xticks(rotation="vertical")
    plt.xlabel("method")
    plt.ylabel("Fidelity")
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.subplots_adjust(bottom=0.35)
    if not os.path.exists("./fidelity_plots/"):
        os.makedirs("./fidelity_plots/")
    plt.savefig("./fidelity_plots/evaluate_fidelity_" + dataset + ".png")
    plt.savefig("./fidelity_plots/evaluate_fidelity_" + dataset + ".pdf")
