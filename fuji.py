import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n========== Network Enrichment Analysis ==========\n")

# =========================================================
# 1. 定义 AAL → Functional Network 映射
# =========================================================

def assign_network(roi_name):

    roi = str(roi_name)

    if ("Cerebelum" in roi) or ("Vermis" in roi):
        return "Cerebellum"

    if any(x in roi for x in [
        "Caudate", "Putamen", "Pallidum", "Thalamus"
    ]):
        return "Subcortical"

    if "Insula" in roi:
        return "Insula_Salience"

    if "Cingulum" in roi:
        return "Cingulate"

    if any(x in roi for x in [
        "Hippocampus", "ParaHippocampal",
        "Amygdala", "Olfactory", "Rectus"
    ]):
        return "Limbic"

    if any(x in roi for x in [
        "Precentral", "Postcentral",
        "Rolandic_Oper", "Supp_Motor_Area",
        "Paracentral_Lobule"
    ]):
        return "Sensorimotor"

    if any(x in roi for x in [
        "Temporal", "Heschl"
    ]):
        return "Temporal"

    if any(x in roi for x in [
        "Calcarine", "Cuneus", "Lingual",
        "Occipital", "Fusiform"
    ]):
        return "Occipital"

    if any(x in roi for x in [
        "Parietal", "SupraMarginal",
        "Angular", "Precuneus"
    ]):
        return "Parietal"

    if "Frontal" in roi:
        return "Frontal"

    return "Other"


# =========================================================
# 2. 读取数据
# =========================================================

print("Loading normative modeling results...")

df = pd.read_csv("step_normative_shared_edges_summary_named.csv")

print("Total edges:", len(df))


# =========================================================
# 3. ROI → Network
# =========================================================

print("\nMapping ROIs to networks...")

df["net_i"] = df["roi_i_name"].apply(assign_network)
df["net_j"] = df["roi_j_name"].apply(assign_network)

# network pair
df["network_pair"] = df.apply(
    lambda x: " -- ".join(sorted([x["net_i"], x["net_j"]])),
    axis=1
)

df.to_csv("step_network_mapped_shared_edges.csv", index=False)

print("Saved: step_network_mapped_shared_edges.csv")


# =========================================================
# 4. 定义分析函数
# =========================================================

def analyze_networks(data, label):

    print(f"\n==============================")
    print(f"===== {label} =====")
    print(f"==============================")

    # -----------------------------------------------------
    # 4.1 Network pair frequency
    # -----------------------------------------------------

    pair_counts = data["network_pair"].value_counts().reset_index()

    pair_counts.columns = [
        "network_pair",
        "count"
    ]

    # -----------------------------------------------------
    # 4.2 Network participation
    # -----------------------------------------------------

    network_counts = pd.concat([
        data["net_i"],
        data["net_j"]
    ]).value_counts().reset_index()

    network_counts.columns = [
        "network",
        "count"
    ]

    # -----------------------------------------------------
    # 4.3 ROI participation (关键新增)
    # -----------------------------------------------------

    roi_counts = pd.concat([
        data["roi_i_name"],
        data["roi_j_name"]
    ]).value_counts().reset_index()

    roi_counts.columns = [
        "roi",
        "count"
    ]

    # -----------------------------------------------------
    # 4.4 ROI → Network
    # -----------------------------------------------------

    roi_counts["network"] = roi_counts["roi"].apply(assign_network)

    # -----------------------------------------------------
    # 4.5 Normalized participation
    # -----------------------------------------------------

    network_roi_number = roi_counts.groupby("network")["roi"].count()

    roi_counts["network_size"] = roi_counts["network"].map(
        network_roi_number
    )

    roi_counts["normalized_count"] = (
        roi_counts["count"] /
        roi_counts["network_size"]
    )

    # -----------------------------------------------------
    # 4.6 保存结果
    # -----------------------------------------------------

    pair_file = f"{label.lower()}_network_pair_counts.csv"

    network_file = f"{label.lower()}_network_counts.csv"

    roi_file = f"{label.lower()}_roi_participation.csv"

    pair_counts.to_csv(pair_file, index=False)

    network_counts.to_csv(network_file, index=False)

    roi_counts.to_csv(roi_file, index=False)

    # -----------------------------------------------------
    # 4.7 输出结果
    # -----------------------------------------------------

    print("\nTop Network Pairs:")
    print(pair_counts.head(10))

    print("\nTop Networks:")
    print(network_counts.head(10))

    print("\nTop ROI Participation:")
    print(roi_counts.head(15))

    return pair_counts, network_counts, roi_counts


# =========================================================
# 5. ASD–AD
# =========================================================

asd_ad_df = df[df["ASD_AD_both_negative"] == True]

asd_ad_pairs, asd_ad_networks, asd_ad_roi = analyze_networks(
    asd_ad_df,
    "ASD_AD"
)


# =========================================================
# 6. AD–FC
# =========================================================

ad_fc_df = df[df["AD_FC_both_negative"] == True]

ad_fc_pairs, ad_fc_networks, ad_fc_roi = analyze_networks(
    ad_fc_df,
    "AD_FC"
)


# =========================================================
# 7. ALL THREE
# =========================================================

all3_df = df[df["ALL_three_negative"] == True]

all3_pairs, all3_networks, all3_roi = analyze_networks(
    all3_df,
    "ALL_THREE"
)


# =========================================================
# 8. 画图函数（保留为辅助统计图）
# =========================================================

def plot_network_pairs(pair_counts, title, filename):

    plt.figure(figsize=(10, 6))

    top = pair_counts.head(10)

    plt.bar(
        top["network_pair"],
        top["count"]
    )

    plt.xticks(rotation=60, ha="right")

    plt.ylabel("Edge count")

    plt.title(title)

    max_count = int(top["count"].max())

    plt.yticks(range(0, max_count + 1))

    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    plt.close()

    print("Saved:", filename)


# =========================================================
# 9. ROI participation visualization（新增）
# =========================================================

def plot_roi_participation(roi_counts, title, filename):

    plt.figure(figsize=(10, 8))

    top = roi_counts.sort_values(
        "normalized_count",
        ascending=False
    ).head(15)

    plt.barh(
        top["roi"],
        top["normalized_count"]
    )

    plt.xlabel("Normalized participation")

    plt.title(title)

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    plt.close()

    print("Saved:", filename)


# =========================================================
# 10. 生成图片
# =========================================================

print("\nGenerating figures...")

# ---------------------------------------------------------
# network pair plots
# ---------------------------------------------------------

plot_network_pairs(
    asd_ad_pairs,
    "ASD–AD shared network pairs",
    "figure_asd_ad_network_pairs.png"
)

plot_network_pairs(
    ad_fc_pairs,
    "AD–FC shared network pairs",
    "figure_ad_fc_network_pairs.png"
)

plot_network_pairs(
    all3_pairs,
    "Overall shared network pairs",
    "figure_all3_network_pairs.png"
)

# ---------------------------------------------------------
# ROI participation plots（新增）
# ---------------------------------------------------------

plot_roi_participation(
    asd_ad_roi,
    "ASD–AD ROI participation",
    "figure_asd_ad_roi_participation.png"
)

plot_roi_participation(
    ad_fc_roi,
    "AD–FC ROI participation",
    "figure_ad_fc_roi_participation.png"
)

plot_roi_participation(
    all3_roi,
    "Overall ROI participation",
    "figure_all3_roi_participation.png"
)


# =========================================================
# 11. 输出总结
# =========================================================

print("\n========== Analysis Finished ==========\n")

print("Saved CSV files:")

print("step_network_mapped_shared_edges.csv")

print("asd_ad_network_pair_counts.csv")
print("asd_ad_network_counts.csv")
print("asd_ad_roi_participation.csv")

print("ad_fc_network_pair_counts.csv")
print("ad_fc_network_counts.csv")
print("ad_fc_roi_participation.csv")

print("all_three_network_pair_counts.csv")
print("all_three_network_counts.csv")
print("all_three_roi_participation.csv")

print("\nSaved Figures:")

print("figure_asd_ad_network_pairs.png")
print("figure_ad_fc_network_pairs.png")
print("figure_all3_network_pairs.png")

print("figure_asd_ad_roi_participation.png")
print("figure_ad_fc_roi_participation.png")
print("figure_all3_roi_participation.png")
