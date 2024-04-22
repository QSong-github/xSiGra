import itertools
import os
import statistics

import pandas as pd

# for dataset in ["lung5-rep1","lung5-rep2","lung5-rep3","lung6","lung9-rep1","lung9-rep2","lung12","lung13"]:
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

    for pair in list(itertools.combinations(list_1, 2)):
        if pair[0] == pair[1]:
            continue
        pair1 = pd.read_csv(
            "../cluster_results_gradcam_gt1/" + dataset + "/" + pair[0], index_col=0
        )
        pair2 = pd.read_csv(
            "../cluster_results_gradcam_gt1/" + dataset + "/" + pair[1], index_col=0
        )

        list1 = list(pair1.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)
        list2 = list(pair2.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        jaccard_distance = 1 - (float(intersection) / union)
        gradcam.append(jaccard_distance)

        pair1 = pd.read_csv(
            "../cluster_results_saliency_gt1/" + dataset + "/" + pair[0], index_col=0
        )
        pair2 = pd.read_csv(
            "../cluster_results_saliency_gt1/" + dataset + "/" + pair[1], index_col=0
        )
        list1 = list(pair1.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)
        list2 = list(pair2.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        jaccard_distance = 1 - (float(intersection) / union)
        saliency.append(jaccard_distance)

        pair1 = pd.read_csv(
            "../cluster_results_inputxgradient_gt1/" + dataset + "/" + pair[0],
            index_col=0,
        )
        pair2 = pd.read_csv(
            "../cluster_results_inputxgradient_gt1/" + dataset + "/" + pair[1],
            index_col=0,
        )
        list1 = list(pair1.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)
        list2 = list(pair2.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        jaccard_distance = 1 - (float(intersection) / union)
        inputxgradient.append(jaccard_distance)

        pair1 = pd.read_csv(
            "../cluster_results_deconvolution_gt1/" + dataset + "/" + pair[0],
            index_col=0,
        )
        pair2 = pd.read_csv(
            "../cluster_results_deconvolution_gt1/" + dataset + "/" + pair[1],
            index_col=0,
        )
        list1 = list(pair1.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)
        list2 = list(pair2.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        jaccard_distance = 1 - (float(intersection) / union)
        deconvolution.append(jaccard_distance)

        pair1 = pd.read_csv(
            "../cluster_results_guidedbackprop_gt1/" + dataset + "/" + pair[0],
            index_col=0,
        )
        pair2 = pd.read_csv(
            "../cluster_results_guidedbackprop_gt1/" + dataset + "/" + pair[1],
            index_col=0,
        )
        list1 = list(pair1.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)
        list2 = list(pair2.iloc[:, 9:].var().sort_values(ascending=False)[:30].index)

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        jaccard_distance = 1 - (float(intersection) / union)
        guidedbackprop.append(jaccard_distance)

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
    plt.ylabel("Contrastivity")
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.subplots_adjust(bottom=0.35)
    if not os.path.exists("./contrastivity_plots/"):
        os.makedirs("./contrastivity_plots/")
    plt.savefig("./contrastivity_plots/evaluate_contrastivity_" + dataset + ".png")
    plt.savefig("./contrastivity_plots/evaluate_contrastivity_" + dataset + ".pdf")
