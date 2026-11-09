import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

# =========================
# 1. 读取数据
# =========================
feat = pd.read_csv("step1_features_6670.csv")
meta = pd.read_csv("step_normative_metadata_clean.csv")

feat["subject_id"] = feat["subject_id"].astype(str)
meta["subject_id"] = meta["subject_id"].astype(str)

data = feat.merge(meta, on=["subject_id", "group"], how="inner")

print("Merged data shape:", data.shape)

# =========================
# 2. 读取共享边
# =========================
asd_ad_edges = pd.read_csv("asd_ad_edges_named.csv")
ad_fc_edges = pd.read_csv("ad_fc_edges_named.csv")

edges_asd_ad = asd_ad_edges["edge"].tolist()
edges_ad_fc = ad_fc_edges["edge"].tolist()

# 合并去重
shared_edges = sorted(list(set(edges_asd_ad + edges_ad_fc)))

print("ASD-AD edges:", len(edges_asd_ad))
print("AD-FC edges:", len(edges_ad_fc))
print("Unique shared edges:", len(shared_edges))

# =========================
# 3. 只保留分析需要的列
# =========================
base_cols = ["subject_id", "dataset", "diagnosis", "age", "age2", "sex01"]
need_cols = base_cols + shared_edges

df = data[need_cols].copy()
df = df.dropna(subset=["age", "age2", "sex01"])

print("Analysis dataframe shape:", df.shape)

# HC 用于拟合 normative model
hc = df[df["diagnosis"] == "HC"].copy()
print("HC sample size:", len(hc))
print(hc.groupby("dataset").size())

# =========================
# 4. 批量拟合每条边
# =========================
summary_rows = []
zdev_rows = []

for edge_col in shared_edges:
    try:
        # 当前边的数据
        tmp = df[base_cols + [edge_col]].dropna().copy()
        hc_tmp = tmp[tmp["diagnosis"] == "HC"].copy()

        # 至少要有一定HC样本
        if len(hc_tmp) < 30:
            print(f"[Skip] {edge_col}: too few HC samples")
            continue

        # normative model
        model = smf.ols(
            formula=f"{edge_col} ~ age + age2 + sex01 + C(dataset)",
            data=hc_tmp
        ).fit()

        # 预测
        tmp["pred"] = model.predict(tmp)

        hc_pred = model.predict(hc_tmp)
        hc_resid = hc_tmp[edge_col] - hc_pred
        sigma = hc_resid.std(ddof=1)

        if pd.isna(sigma) or sigma == 0:
            print(f"[Skip] {edge_col}: invalid sigma")
            continue

        tmp["z_dev"] = (tmp[edge_col] - tmp["pred"]) / sigma
        tmp["edge"] = edge_col

        zdev_rows.append(
            tmp[["subject_id", "dataset", "diagnosis", "edge", "z_dev"]].copy()
        )

        # 各组统计
        group_stats = tmp.groupby(["dataset", "diagnosis"])["z_dev"].agg(["mean", "std", "count"])

        def get_stat(dataset, diagnosis, stat):
            try:
                return group_stats.loc[(dataset, diagnosis), stat]
            except:
                return np.nan

        # 各关键组均值
        asd_mean = get_stat("ASD", "ASD", "mean")
        asd_hc_mean = get_stat("ASD", "HC", "mean")

        ad_mean = get_stat("AD", "AD", "mean")
        ad_hc_mean = get_stat("AD", "HC", "mean")

        mci_mean = get_stat("AD", "MCI", "mean")

        fc_mean = get_stat("FC", "FC", "mean")
        fc_hc_mean = get_stat("FC", "HC", "mean")

        # 单样本 t 检验：疾病组 z_dev 是否显著偏离 0
        def one_sample_test(dataset, diagnosis):
            vals = tmp[(tmp["dataset"] == dataset) & (tmp["diagnosis"] == diagnosis)]["z_dev"].dropna()
            if len(vals) < 3:
                return np.nan, np.nan
            t, p = ttest_1samp(vals, popmean=0.0)
            return t, p

        asd_t, asd_p = one_sample_test("ASD", "ASD")
        ad_t, ad_p = one_sample_test("AD", "AD")
        mci_t, mci_p = one_sample_test("AD", "MCI")
        fc_t, fc_p = one_sample_test("FC", "FC")

        # 方向判定
        asd_neg = pd.notna(asd_mean) and (asd_mean < 0)
        ad_neg = pd.notna(ad_mean) and (ad_mean < 0)
        fc_neg = pd.notna(fc_mean) and (fc_mean < 0)

        # 属于哪类共享边
        in_asd_ad = edge_col in edges_asd_ad
        in_ad_fc = edge_col in edges_ad_fc

        summary_rows.append({
            "edge": edge_col,
            "in_ASD_AD": in_asd_ad,
            "in_AD_FC": in_ad_fc,

            "hc_sigma": sigma,
            "model_r2": model.rsquared,
            "model_f_pvalue": model.f_pvalue,

            "ASD_mean_z": asd_mean,
            "ASD_t": asd_t,
            "ASD_p": asd_p,

            "AD_mean_z": ad_mean,
            "AD_t": ad_t,
            "AD_p": ad_p,

            "MCI_mean_z": mci_mean,
            "MCI_t": mci_t,
            "MCI_p": mci_p,

            "FC_mean_z": fc_mean,
            "FC_t": fc_t,
            "FC_p": fc_p,

            "ASD_negative": asd_neg,
            "AD_negative": ad_neg,
            "FC_negative": fc_neg,

            "ASD_AD_both_negative": in_asd_ad and asd_neg and ad_neg,
            "AD_FC_both_negative": in_ad_fc and ad_neg and fc_neg,
            "ALL_three_negative": asd_neg and ad_neg and fc_neg,
        })

    except Exception as e:
        print(f"[Error] {edge_col}: {e}")

# =========================
# 5. 保存结果
# =========================
summary_df = pd.DataFrame(summary_rows)

if len(zdev_rows) > 0:
    zdev_df = pd.concat(zdev_rows, axis=0, ignore_index=True)
else:
    zdev_df = pd.DataFrame()

summary_df.to_csv("step_normative_shared_edges_summary.csv", index=False)
zdev_df.to_csv("step_normative_shared_edges_zdev_long.csv", index=False)

print("\nSaved:")
print(" - step_normative_shared_edges_summary.csv")
print(" - step_normative_shared_edges_zdev_long.csv")

# =========================
# 6. 汇总输出
# =========================
print("\n===== Summary =====")
print("Total modeled edges:", len(summary_df))

print("\nASD-AD edges still both negative:",
      summary_df["ASD_AD_both_negative"].sum())

print("AD-FC edges still both negative:",
      summary_df["AD_FC_both_negative"].sum())

print("All-three negative:",
      summary_df["ALL_three_negative"].sum())

print("\nTop 10 edges by strongest AD negative deviation:")
print(summary_df.sort_values("AD_mean_z").head(10)[
    ["edge", "in_ASD_AD", "in_AD_FC", "ASD_mean_z", "AD_mean_z", "FC_mean_z"]
])

print("\nTop 10 edges by strongest FC negative deviation:")
print(summary_df.sort_values("FC_mean_z").head(10)[
    ["edge", "in_ASD_AD", "in_AD_FC", "ASD_mean_z", "AD_mean_z", "FC_mean_z"]
])
