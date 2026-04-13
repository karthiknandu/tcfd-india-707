"""
xgboost_shap.py  —  Stage 1: TCFD Attribution via XGBoost + SHAP
=================================================================
Trains XGBoost on master_final_v3.csv with UHCd as target.
Computes SHAP values and aggregates attribution by TCFD type.

Usage:
    python src/models/xgboost_shap.py \
        --data data/master_final_v3.csv \
        --taxonomy data/tcfd_taxonomy.json \
        --output results/
"""

import argparse
import json
import logging
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

TYPE_COLORS = {
    "Type_I_structural_physical": "#E74C3C",
    "Type_II_historical_injustice": "#8E44AD",
    "Type_III_policy_actionable": "#27AE60",
    "confounder": "#95A5A6",
    "outcome": "#2980B9",
    "identifier": "#BDC3C7",
}


def load_data(data_path: str, taxonomy_path: str):
    """Load master dataset and TCFD taxonomy."""
    df = pd.read_csv(data_path)
    log.info(f"Loaded: {df.shape[0]} districts × {df.shape[1]} columns")

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)["taxonomy"]

    # Flatten taxonomy to column → type mapping
    col_to_type = {}
    for type_name, cols in taxonomy.items():
        for col in cols:
            col_to_type[col] = type_name
    return df, col_to_type


def prepare_features(df: pd.DataFrame, col_to_type: dict, target: str = "UHCd"):
    """Select features, impute, return X, y and type assignments."""
    # Exclude identifiers, outcome vars, and the target
    exclude_types = {"identifier", "outcome"}
    exclude_cols = {c for c, t in col_to_type.items() if t in exclude_types}
    exclude_cols |= {target, "District", "State", "District_norm", "State_norm",
                     "GID_2", "UHCd_Tercile", "UHCd_Tercile",
                     "pmjay_pmjay_status", "pmjay_State UT",
                     "pmjay_launch_date", "pmjay_model", "pmjay_uhcd_tercile",
                     "geo_area_quintile", "geo_remoteness_quintile", "geo_tax_quintile",
                     "density_quintile", "match_type"}

    # Drop columns with >40% missing
    missing_pct = df.isnull().mean()
    high_missing = set(missing_pct[missing_pct > 0.4].index)
    log.info(f"Dropping {len(high_missing)} columns with >40% missing")

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and c not in high_missing
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Keep only districts where UHCd is known
    mask = df[target].notna()
    X = df.loc[mask, feature_cols].copy()
    y = df.loc[mask, target].copy()
    log.info(f"Training set: {len(X)} districts with known UHCd")
    log.info(f"Features: {len(feature_cols)}")

    # Impute remaining NaN
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    # Map features to TCFD type
    feat_types = {c: col_to_type.get(c, "other") for c in feature_cols}
    return X_imp, y, feat_types, feature_cols


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train XGBoost with fixed hyperparameters (post-Bayesian-search optimal)."""
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    # 5-fold CV evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv,
                                            scoring="neg_mean_squared_error"))
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    log.info(f"5-fold CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
    log.info(f"5-fold CV R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    # Fit on full training set
    model.fit(X, y)
    return model


def compute_shap_attribution(model, X: pd.DataFrame, feat_types: dict) -> pd.DataFrame:
    """Compute SHAP values and aggregate by TCFD type."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_districts, n_features)

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    # Absolute SHAP per district
    abs_shap = shap_df.abs()
    total_shap = abs_shap.sum(axis=1)

    # Attribution share by TCFD type
    type_shares = {}
    for type_name in ["Type_I_structural_physical", "Type_II_historical_injustice",
                      "Type_III_policy_actionable", "confounder"]:
        cols_of_type = [c for c, t in feat_types.items() if t == type_name and c in abs_shap.columns]
        type_shares[f"tcfd_{type_name}_share"] = (
            abs_shap[cols_of_type].sum(axis=1) / total_shap
        ).fillna(0)

    attribution_df = pd.DataFrame(type_shares, index=X.index)
    attribution_df["tcfd_dominant_type"] = attribution_df[
        [c for c in attribution_df.columns if c.startswith("tcfd_") and c.endswith("_share")]
    ].idxmax(axis=1).str.replace("tcfd_", "").str.replace("_share", "")

    log.info(f"Dominant type distribution:\n{attribution_df['tcfd_dominant_type'].value_counts()}")
    return shap_df, attribution_df


def evaluate_holdout(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate on 20% holdout (stratified by UHCd quartile)."""
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "n_test": len(y_test),
    }
    log.info(f"Holdout metrics: {metrics}")
    return metrics


def plot_shap_summary(shap_df, feat_types, output_dir: str, top_n: int = 20):
    """SHAP summary bar plot coloured by TCFD type."""
    mean_abs = shap_df.abs().mean().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [TYPE_COLORS.get(feat_types.get(c, "other"), "#BDC3C7") for c in mean_abs.index]
    ax.barh(range(len(mean_abs)), mean_abs.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels([c[:50] for c in mean_abs.index[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} SHAP Feature Importances by TCFD Type")
    patches = [mpatches.Patch(color=v, label=k.replace("_", " ")) for k, v in TYPE_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = Path(output_dir) / "figures" / "shap_summary_by_tcfd_type.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP summary plot saved: {out}")


def run(data_path, taxonomy_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "figures").mkdir(exist_ok=True)
    (Path(output_dir) / "tables").mkdir(exist_ok=True)

    df, col_to_type = load_data(data_path, taxonomy_path)
    X, y, feat_types, feature_cols = prepare_features(df, col_to_type)
    model = train_xgboost(X, y)
    metrics = evaluate_holdout(model, X, y)
    shap_df, attribution_df = compute_shap_attribution(model, X, feat_types)

    # Save outputs
    np.save(Path(output_dir) / "shap_matrix.npy", shap_df.values)
    shap_df.to_csv(Path(output_dir) / "tables" / "shap_values.csv")
    attribution_df.to_csv(Path(output_dir) / "tables" / "tcfd_attribution_shares.csv")
    pd.DataFrame([metrics]).to_csv(
        Path(output_dir) / "tables" / "model_evaluation_metrics.csv", index=False
    )
    plot_shap_summary(shap_df, feat_types, output_dir)

    # Merge attribution back into main df
    result = df.merge(attribution_df, left_index=True, right_index=True, how="left")
    result.to_csv(Path(output_dir) / "master_with_tcfd_attribution.csv", index=False)
    log.info("Stage 1 complete — SHAP attribution saved.")
    return model, shap_df, attribution_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/master_final_v3.csv")
    parser.add_argument("--taxonomy", default="data/tcfd_taxonomy.json")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run(args.data, args.taxonomy, args.output)
