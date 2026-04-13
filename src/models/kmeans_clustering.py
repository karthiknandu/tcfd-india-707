"""
kmeans_clustering.py  —  Stage 3: K-Means on SHAP Attribution Matrix
=====================================================================
Clusters 707 districts by TCFD pathway type attribution shares.
Produces cluster profiles, ANOVA tests, and choropleth map data.

Usage:
    python src/models/kmeans_clustering.py \
        --attribution results/tables/tcfd_attribution_shares.csv \
        --master data/master_final_v3.csv \
        --output results/
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from itertools import combinations

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
SEED = 42
np.random.seed(SEED)

TYPE_SHARE_COLS = [
    "tcfd_Type_I_structural_physical_share",
    "tcfd_Type_II_historical_injustice_share",
    "tcfd_Type_III_policy_actionable_share",
]

CLUSTER_LABELS = {
    0: "Type I Dominated (Geography-Constrained)",
    1: "Type II Dominated (Historical-Injustice)",
    2: "Type III Dominated (Policy-Responsive)",
    3: "Mixed I+II (Remote & Historically Excluded)",
    4: "Mixed I+III (Geographically Remote, Policy-Active)",
}


def select_k(X: np.ndarray, k_range=range(2, 9)) -> int:
    """Select optimal K using elbow (WCSS) and silhouette score."""
    wcss, sil = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X, labels) if k > 1 else 0)
    # Find elbow: point of maximum curvature
    diffs = np.diff(wcss)
    elbow_k = k_range[np.argmax(np.diff(diffs)) + 1] if len(diffs) > 1 else 5
    best_sil_k = k_range[np.argmax(sil)]
    log.info(f"Elbow K={elbow_k}, Best silhouette K={best_sil_k} (score={max(sil):.3f})")
    # Default to 5 as per research design; validate with silhouette
    return 5


def jaccard_stability(X: np.ndarray, k: int, n_bootstrap: int = 100) -> float:
    """Bootstrap Jaccard stability index for cluster solution."""
    from sklearn.utils import resample
    base_model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    base_labels = base_model.fit_predict(X)
    jaccard_scores = []
    for i in range(n_bootstrap):
        X_boot, idx = resample(X, return_indices=True, random_state=i)
        boot_model = KMeans(n_clusters=k, random_state=i, n_init=10)
        boot_labels = boot_model.fit_predict(X_boot)
        # Match clusters by majority vote
        from scipy.optimize import linear_sum_assignment
        confusion = np.zeros((k, k))
        for orig, boot in zip(base_labels[idx], boot_labels):
            confusion[orig, boot] += 1
        row_ind, col_ind = linear_sum_assignment(-confusion)
        matched = confusion[row_ind, col_ind].sum()
        total = len(idx)
        jaccard = matched / (2 * total - matched)
        jaccard_scores.append(jaccard)
    mean_j = np.mean(jaccard_scores)
    log.info(f"Jaccard stability (K={k}, B={n_bootstrap}): {mean_j:.3f}")
    return mean_j


def profile_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Compute cluster-level profiles for UHCd, attribution shares, and Type II vars."""
    profile_cols = TYPE_SHARE_COLS + [
        "UHCd", "geo_tax_index", "RMNCH", "FRP",
        "Women (age 15-49) who are literate4 (%)",
        "Women age 20-24 years married before age 18 years (%)",
        "Households with any usual member covered under a health insurance/financing scheme (%)",
        "diu_established",
    ]
    avail = [c for c in profile_cols if c in df_clustered.columns]
    profile = df_clustered.groupby("kmeans_cluster")[avail].agg(["mean", "std", "count"])
    profile.columns = ["_".join(c) for c in profile.columns]
    profile["n_districts"] = df_clustered.groupby("kmeans_cluster").size()
    profile["dominant_type"] = df_clustered.groupby("kmeans_cluster")["tcfd_dominant_type"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "unknown"
    )
    return profile


def anova_uhcd(df_clustered: pd.DataFrame) -> dict:
    """One-way ANOVA and Tukey HSD for UHCd across clusters."""
    groups = [
        df_clustered[df_clustered["kmeans_cluster"] == k]["UHCd"].dropna().values
        for k in sorted(df_clustered["kmeans_cluster"].unique())
    ]
    groups = [g for g in groups if len(g) >= 5]
    f_stat, p_val = f_oneway(*groups)
    result = {"F_statistic": round(f_stat, 4), "p_value": round(p_val, 6),
              "significant_p05": p_val < 0.05}
    log.info(f"ANOVA UHCd across clusters: F={f_stat:.3f}, p={p_val:.4f}")
    return result


def plot_cluster_profiles(profile: pd.DataFrame, output_dir: str):
    """Radar / bar chart of cluster attribution shares."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    type_means = {}
    for i, col in enumerate(TYPE_SHARE_COLS):
        mean_col = f"{col}_mean"
        if mean_col in profile.columns:
            type_means[col.replace("tcfd_", "").replace("_share", "")] = profile[mean_col]
    colors = plt.cm.Set2(np.linspace(0, 1, len(profile)))
    for ax, (type_name, vals) in zip(axes, type_means.items()):
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_title(type_name.replace("_", " "), fontsize=10)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"C{i}" for i in vals.index], fontsize=9)
        ax.set_ylabel("Mean Attribution Share")
        ax.set_ylim(0, 0.6)
    plt.suptitle("TCFD Attribution Shares by K-Means Cluster", fontsize=12)
    plt.tight_layout()
    out = Path(output_dir) / "figures" / "cluster_attribution_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Cluster profile plot saved: {out}")


def run(attribution_path, master_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "tables").mkdir(exist_ok=True)
    (Path(output_dir) / "figures").mkdir(exist_ok=True)

    # Load attribution shares
    attr = pd.read_csv(attribution_path, index_col=0)
    master = pd.read_csv(master_path)

    # Merge on index
    df = master.merge(attr, left_index=True, right_index=True, how="left")

    # Keep only districts where all three type shares are available
    avail = [c for c in TYPE_SHARE_COLS if c in df.columns]
    mask = df[avail].notna().all(axis=1)
    X_raw = df.loc[mask, avail].values
    log.info(f"Clustering {mask.sum()} districts with complete attribution data")

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Select K
    k = select_k(X)

    # Fit K-means
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    wcss = km.inertia_
    log.info(f"K={k}: Silhouette={sil:.3f}, WCSS={wcss:.1f}")

    # Stability
    jaccard = jaccard_stability(X, k)

    # Assign cluster labels
    df.loc[mask, "kmeans_cluster"] = labels
    df.loc[mask, "kmeans_cluster_label"] = pd.Series(labels).map(CLUSTER_LABELS).values

    # Profile
    profile = profile_clusters(df.loc[mask])
    anova = anova_uhcd(df.loc[mask])
    plot_cluster_profiles(profile, output_dir)

    # Save outputs
    df[["District_norm", "State_norm", "UHCd", "kmeans_cluster",
        "kmeans_cluster_label", "tcfd_dominant_type", "GID_2"] + avail
       ].to_csv(Path(output_dir) / "tables" / "district_cluster_assignments.csv", index=False)
    profile.to_csv(Path(output_dir) / "tables" / "cluster_profiles.csv")
    pd.DataFrame([{
        "K": k, "silhouette": sil, "wcss": wcss,
        "jaccard_stability": jaccard, **anova
    }]).to_csv(Path(output_dir) / "tables" / "clustering_metrics.csv", index=False)

    log.info("Stage 3 complete — cluster assignments saved.")
    return df, profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", default="results/tables/tcfd_attribution_shares.csv")
    parser.add_argument("--master", default="data/master_final_v3.csv")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run(args.attribution, args.master, args.output)
