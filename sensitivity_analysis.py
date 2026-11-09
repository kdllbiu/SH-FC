import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load results
# ==========================

asd = pd.read_csv("step3_ASD_vs_HC_edge_stats.csv")
ad  = pd.read_csv("step3_AD_vs_HC_edge_stats.csv")
fc  = pd.read_csv("step3_FC_vs_HC_edge_stats.csv")

# ==========================
# Helper
# ==========================

def jaccard(a, b):
    union = len(a | b)
    if union == 0:
        return np.nan
    return len(a & b) / union


# ==========================
# Part 1
# Top-k overlap robustness
# ==========================

k_list = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20
]

pairs = {
    "ASD-AD": (asd, ad),
    "AD-FC": (ad, fc),
    "ASD-FC": (asd, fc)
}

overlap_results = []

for pair_name, (r1, r2) in pairs.items():

    df = r1[["edge","hedges_g"]].merge(
        r2[["edge","hedges_g"]],
        on="edge",
        suffixes=("_1","_2")
    )

    df["abs1"] = df["hedges_g_1"].abs()
    df["abs2"] = df["hedges_g_2"].abs()

    n = len(df)

    for frac in k_list:

        k = int(round(frac*n))

        top1 = set(
            df.nlargest(k,"abs1")["edge"]
        )

        top2 = set(
            df.nlargest(k,"abs2")["edge"]
        )

        overlap = len(top1 & top2)

        overlap_results.append({
            "pair": pair_name,
            "k": frac,
            "overlap": overlap,
            "jaccard": jaccard(top1, top2)
        })

overlap_df = pd.DataFrame(overlap_results)

# ==========================
# Plot S1A
# ==========================

plt.figure(figsize=(7,5))

for pair in overlap_df["pair"].unique():

    sub = overlap_df[
        overlap_df["pair"]==pair
    ]

    plt.plot(
        sub["k"]*100,
        sub["jaccard"],
        marker="o",
        label=pair
    )

plt.xlabel("Top-k edges (%)")
plt.ylabel("Jaccard overlap")
plt.title("Robustness of Top-k Overlap Across Edge Selection Thresholds")
plt.legend()

plt.tight_layout()
plt.savefig(
    "Fig_S1A_topk_overlap.png",
    dpi=300
)

plt.close()


# ==========================
# Part 2
# Sign concordance
# ==========================

sign_results = []

for pair_name, (r1, r2) in pairs.items():

    df = r1[["edge","hedges_g"]].merge(
        r2[["edge","hedges_g"]],
        on="edge",
        suffixes=("_1","_2")
    )

    df["abs1"] = df["hedges_g_1"].abs()
    df["abs2"] = df["hedges_g_2"].abs()

    n = len(df)

    for frac in k_list:

        k = int(round(frac*n))

        top1 = set(
            df.nlargest(k,"abs1")["edge"]
        )

        top2 = set(
            df.nlargest(k,"abs2")["edge"]
        )

        shared = df[
            df["edge"].isin(top1 & top2)
        ]

        if len(shared)==0:

            concordance=np.nan

        else:

            concordance=np.mean(
                np.sign(shared["hedges_g_1"])
                ==
                np.sign(shared["hedges_g_2"])
            )

        sign_results.append({
            "pair": pair_name,
            "k": frac,
            "concordance": concordance
        })

sign_df = pd.DataFrame(sign_results)

# ==========================
# Plot S1B
# ==========================

plt.figure(figsize=(7,5))

for pair in sign_df["pair"].unique():

    sub = sign_df[
        sign_df["pair"]==pair
    ]

    plt.plot(
        sub["k"]*100,
        sub["concordance"],
        marker="o",
        label=pair
    )

plt.xlabel("Top-k edges (%)")
plt.ylabel("Sign concordance")
plt.title("Consistency of Effect Direction Across Shared Edges")
plt.legend()

plt.tight_layout()
plt.savefig(
    "Fig_S1B_sign_concordance.png",
    dpi=300
)

plt.close()


# ==========================
# Part 3
# FC threshold sensitivity
# ==========================

thresholds = [
    0.05,
    0.01,
    0.005
]

rows = []

asd_sig = set(
    asd[
        asd["p_fdr"]<0.05
    ]["edge"]
)

ad_sig = set(
    ad[
        ad["p_fdr"]<0.05
    ]["edge"]
)

for thr in thresholds:

    fc_sig = set(
        fc[
            fc["p"]<thr
        ]["edge"]
    )

    asd_fc = len(
        asd_sig & fc_sig
    )

    ad_fc = len(
        ad_sig & fc_sig
    )

    triple = len(
        asd_sig &
        ad_sig &
        fc_sig
    )

    rows.append({
        "threshold":thr,
        "ASD-FC":asd_fc,
        "AD-FC":ad_fc,
        "Triple":triple
    })

thr_df = pd.DataFrame(rows)

thr_df.to_csv(
    "FC_threshold_sensitivity.csv",
    index=False
)

# ==========================
# Plot S1C
# ==========================

plt.figure(figsize=(7,5))

plt.plot(
    thr_df["threshold"],
    thr_df["ASD-FC"],
    marker="o",
    label="ASD-FC"
)

plt.plot(
    thr_df["threshold"],
    thr_df["AD-FC"],
    marker="o",
    label="AD-FC"
)

plt.plot(
    thr_df["threshold"],
    thr_df["Triple"],
    marker="o",
    label="Triple"
)

plt.gca().invert_xaxis()

plt.xlabel("FC threshold")
plt.ylabel("Shared edges")
plt.title("Threshold Sensitivity")

plt.legend()

plt.tight_layout()

plt.savefig(
    "Fig_S1C_fc_threshold_sensitivity.png",
    dpi=300
)

plt.close()

print()
print("Finished.")
print()

print("Generated:")
print("Fig_S1A_topk_overlap.png")
print("Fig_S1B_sign_concordance.png")
print("Fig_S1C_fc_threshold_sensitivity.png")
print("FC_threshold_sensitivity.csv")
