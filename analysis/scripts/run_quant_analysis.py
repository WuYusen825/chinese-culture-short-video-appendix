#!/usr/bin/env python3
"""Reproducible quantitative analysis for the short-video survey project.

Inputs are read from the project root. Original files are never modified.
Outputs are written under final_report_project/analysis/.
"""

from __future__ import annotations

import itertools
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "final_report_project" / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"

SECOND_ROUND_XLSX = PROJECT_ROOT / "第二轮调研.xlsx"
FIRST_ROUND_XLSX = PROJECT_ROOT / "第一轮调研.xlsx"
SECOND_ROUND_PDF = PROJECT_ROOT / "第二轮问卷.pdf"

DIMENSIONS = {
    "Attraction": [f"Attraction_{i}" for i in range(1, 5)],
    "Resonance": [f"Resonance_{i}" for i in range(1, 6)],
    "Retention": [f"Retention_{i}" for i in range(1, 6)],
    "Conversion": [f"Conversion_{i}" for i in range(1, 6)],
}
DIMENSION_ORDER = list(DIMENSIONS)
ITEM_COLUMNS = [item for items in DIMENSIONS.values() for item in items]
SOURCE_DEMOGRAPHIC_COLUMNS = ["ID", "Sexuality", "Continent"]
DEMOGRAPHIC_COLUMNS = ["ID", "Gender", "Continent"]

DIMENSION_DESCRIPTIONS = {
    "Attraction": "First-sight visual/audio attention to Chinese-culture short videos.",
    "Resonance": "Emotional, cognitive, and value-based resonance with Chinese-culture short videos.",
    "Retention": "Longer-term stickiness: following, sharing, commenting, and creator tracking.",
    "Conversion": "Behavioral conversion: searching, buying, imitating, app use, and offline action.",
}

ITEM_LABELS = {
    "Attraction_1": "Visual strikingness and curiosity from unfamiliar Chinese cultural elements.",
    "Attraction_2": "Music or traditional sound effects compelling continued viewing.",
    "Attraction_3": "Large-scale traditional cultural displays stopping scrolling.",
    "Attraction_4": "Modern Chinese city visuals attracting attention.",
    "Resonance_1": "Being moved by scenes or stories in Chinese short videos.",
    "Resonance_2": "Chinese short videos breaking previous stereotypes about China.",
    "Resonance_3": "Universal values in videos striking a chord.",
    "Resonance_4": "Lifestyle scenes inspiring personal plans or reflections.",
    "Resonance_5": "Willingness to search for stories behind memorable people or events.",
    "Retention_1": "Clicking creator profile to watch more after an interesting video.",
    "Retention_2": "Sharing Chinese-culture short videos with friends or on social media.",
    "Retention_3": "Actively searching for updates from a liked Chinese creator.",
    "Retention_4": "Reading or joining comments to understand cultural background.",
    "Retention_5": "Turning on post notifications for a Chinese creator.",
    "Conversion_1": "Searching specific keywords from videos on external platforms.",
    "Conversion_2": "Purchasing Chinese-related products because of short videos.",
    "Conversion_3": "Practicing or imitating content from Chinese-culture videos.",
    "Conversion_4": "Downloading Chinese local social apps for authentic content.",
    "Conversion_5": "Taking offline action after watching Chinese-culture videos.",
}

SCORE_NOTE = (
    "Items are treated as five-point Likert-scale responses; higher values indicate "
    "stronger or more positive engagement within the named construct."
)


@dataclass
class SavedOutputs:
    tables: list[Path]
    figures: list[Path]
    data: list[Path]


def ensure_directories() -> None:
    for directory in [ANALYSIS_DIR, TABLES_DIR, FIGURES_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_second_round() -> pd.DataFrame:
    df = pd.read_excel(SECOND_ROUND_XLSX)
    df = df.dropna(how="all").copy()
    missing_columns = [c for c in SOURCE_DEMOGRAPHIC_COLUMNS + ITEM_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in second-round data: {missing_columns}")
    df = df[SOURCE_DEMOGRAPHIC_COLUMNS + ITEM_COLUMNS].copy()
    df = df.rename(columns={"Sexuality": "Gender"})
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Continent"] = df["Continent"].astype(str).str.strip()
    for col in ITEM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_and_score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = []
    cleaned = df.copy()
    for col in ITEM_COLUMNS:
        values = cleaned[col].dropna()
        outside = values[~values.between(1, 5)]
        checks.append(
            {
                "variable": col,
                "missing_n": int(cleaned[col].isna().sum()),
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
                "outside_expected_scale_n": int(outside.shape[0]),
            }
        )
    checks.extend(
        [
            {
                "variable": "ID",
                "missing_n": int(cleaned["ID"].isna().sum()),
                "min": float(cleaned["ID"].min()),
                "max": float(cleaned["ID"].max()),
                "outside_expected_scale_n": int(cleaned["ID"].duplicated().sum()),
            },
            {
                "variable": "Gender",
                "missing_n": int(cleaned["Gender"].replace("", np.nan).isna().sum()),
                "min": np.nan,
                "max": np.nan,
                "outside_expected_scale_n": 0,
            },
            {
                "variable": "Continent",
                "missing_n": int(cleaned["Continent"].replace("", np.nan).isna().sum()),
                "min": np.nan,
                "max": np.nan,
                "outside_expected_scale_n": 0,
            },
        ]
    )
    for dim, cols in DIMENSIONS.items():
        cleaned[f"{dim}_score"] = cleaned[cols].mean(axis=1)
        cleaned[f"{dim}_valid_items"] = cleaned[cols].notna().sum(axis=1)
    cleaned["Engagement_overall_score"] = cleaned[[f"{d}_score" for d in DIMENSION_ORDER]].mean(axis=1)
    return cleaned, pd.DataFrame(checks)


def cronbach_alpha(data: pd.DataFrame) -> float:
    x = data.dropna(axis=0, how="any")
    k = x.shape[1]
    if k <= 1 or x.shape[0] <= 1:
        return np.nan
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float(k / (k - 1) * (1 - item_var / total_var))


def ci95(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().to_numpy(dtype=float)
    n = values.size
    if n <= 1:
        return np.nan, np.nan
    mean = values.mean()
    se = values.std(ddof=1) / math.sqrt(n)
    critical = stats.t.ppf(0.975, df=n - 1)
    return float(mean - critical * se), float(mean + critical * se)


def cohen_d_independent(a: Iterable[float], b: Iterable[float]) -> float:
    a = pd.Series(a).dropna().to_numpy(dtype=float)
    b = pd.Series(b).dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def paired_d(a: Iterable[float], b: Iterable[float]) -> float:
    diff = pd.Series(a).reset_index(drop=True) - pd.Series(b).reset_index(drop=True)
    diff = diff.dropna().to_numpy(dtype=float)
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return np.nan
    return float(diff.mean() / diff.std(ddof=1))


def eta_squared_oneway(groups: list[np.ndarray]) -> float:
    groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
    if not groups:
        return np.nan
    all_values = np.concatenate(groups)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_values - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def make_codebook() -> pd.DataFrame:
    rows = [
        {
            "variable": "ID",
            "construct": "Identifier",
            "item_no": "",
            "description": "Respondent identifier from the working Excel file.",
            "type": "numeric identifier",
            "score_direction": "Not a scale score",
        },
        {
            "variable": "Gender",
            "construct": "Demographic",
            "item_no": "",
            "description": "Self-reported gender category retained in the working file.",
            "type": "categorical",
            "score_direction": "Female/Male groups in this dataset",
        },
        {
            "variable": "Continent",
            "construct": "Demographic",
            "item_no": "",
            "description": "Respondent continent category retained in the working file.",
            "type": "categorical",
            "score_direction": "Asia, Europe, North America, South America, Africa in this dataset",
        },
    ]
    for dim, items in DIMENSIONS.items():
        for idx, item in enumerate(items, start=1):
            rows.append(
                {
                    "variable": item,
                    "construct": dim,
                    "item_no": idx,
                    "description": ITEM_LABELS.get(item, ""),
                    "type": "Likert-type numeric item",
                    "score_direction": "Five-point Likert scale; higher = stronger/more positive engagement",
                }
            )
    for dim in DIMENSION_ORDER:
        rows.append(
            {
                "variable": f"{dim}_score",
                "construct": dim,
                "item_no": "mean",
                "description": DIMENSION_DESCRIPTIONS[dim],
                "type": "Scale score",
                "score_direction": "Mean of dimension items; higher = stronger construct score",
            }
        )
    rows.append(
        {
            "variable": "Engagement_overall_score",
            "construct": "Overall",
            "item_no": "mean",
            "description": "Mean of the four dimension scores.",
            "type": "Scale score",
            "score_direction": "Higher = stronger overall engagement",
        }
    )
    return pd.DataFrame(rows)


def sample_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["Gender", "Continent"]:
        counts = df[var].value_counts(dropna=False).sort_index()
        for category, n in counts.items():
            rows.append(
                {
                    "variable": var,
                    "category": category,
                    "n": int(n),
                    "percent": float(n / len(df) * 100),
                }
            )
    return pd.DataFrame(rows)


def item_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ITEM_COLUMNS:
        values = df[col].dropna()
        low, high = ci95(values)
        rows.append(
            {
                "variable": col,
                "construct": col.split("_")[0],
                "n": int(values.shape[0]),
                "missing_n": int(df[col].isna().sum()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_low": low,
                "ci95_high": high,
                "skew": float(stats.skew(values, bias=False)),
                "kurtosis": float(stats.kurtosis(values, bias=False)),
            }
        )
    return pd.DataFrame(rows)


def dimension_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in DIMENSION_ORDER + ["Engagement_overall"]:
        score = f"{dim}_score" if dim in DIMENSION_ORDER else "Engagement_overall_score"
        values = df[score].dropna()
        low, high = ci95(values)
        rows.append(
            {
                "dimension": dim,
                "n": int(values.shape[0]),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_low": low,
                "ci95_high": high,
                "skew": float(stats.skew(values, bias=False)),
                "kurtosis": float(stats.kurtosis(values, bias=False)),
            }
        )
    return pd.DataFrame(rows)


def reliability_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    item_rows = []
    construct_sets = {**DIMENSIONS, "Overall_19_items": ITEM_COLUMNS}
    for construct, cols in construct_sets.items():
        alpha = cronbach_alpha(df[cols])
        rows.append(
            {
                "construct": construct,
                "n_items": len(cols),
                "n_complete": int(df[cols].dropna().shape[0]),
                "cronbach_alpha": alpha,
            }
        )
        for item in cols:
            other_cols = [c for c in cols if c != item]
            total_without = df[other_cols].sum(axis=1)
            item_total = df[item].corr(total_without)
            alpha_deleted = cronbach_alpha(df[other_cols]) if len(other_cols) > 1 else np.nan
            item_rows.append(
                {
                    "construct": construct,
                    "item": item,
                    "item_mean": float(df[item].mean()),
                    "item_sd": float(df[item].std(ddof=1)),
                    "corrected_item_total_r": float(item_total),
                    "alpha_if_deleted": alpha_deleted,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(item_rows)


def dimension_difference_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [f"{d}_score" for d in DIMENSION_ORDER]
    complete = df[score_cols].dropna()
    friedman = stats.friedmanchisquare(*(complete[col] for col in score_cols))
    n = complete.shape[0]
    k = len(score_cols)
    kendalls_w = friedman.statistic / (n * (k - 1)) if n > 0 else np.nan
    overall = pd.DataFrame(
        [
            {
                "test": "Friedman test across four within-subject dimensions",
                "n": int(n),
                "k_dimensions": int(k),
                "statistic_chi2": float(friedman.statistic),
                "df": int(k - 1),
                "p_value": float(friedman.pvalue),
                "effect_kendalls_w": float(kendalls_w),
            }
        ]
    )

    rows = []
    for dim_a, dim_b in itertools.combinations(DIMENSION_ORDER, 2):
        a = df[f"{dim_a}_score"]
        b = df[f"{dim_b}_score"]
        pair = pd.concat([a, b], axis=1).dropna()
        diff = pair.iloc[:, 0] - pair.iloc[:, 1]
        try:
            wilcoxon = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="wilcox")
            w_stat = float(wilcoxon.statistic)
            p_value = float(wilcoxon.pvalue)
        except ValueError:
            w_stat = np.nan
            p_value = np.nan
        rows.append(
            {
                "dimension_a": dim_a,
                "dimension_b": dim_b,
                "n_pairs": int(pair.shape[0]),
                "mean_a": float(pair.iloc[:, 0].mean()),
                "mean_b": float(pair.iloc[:, 1].mean()),
                "mean_difference_a_minus_b": float(diff.mean()),
                "wilcoxon_w": w_stat,
                "p_value": p_value,
                "paired_cohen_dz": paired_d(pair.iloc[:, 0], pair.iloc[:, 1]),
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["p_holm"] = multipletests(pairwise["p_value"].fillna(1), method="holm")[1]
    return overall, pairwise


def gender_differences(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    groups = sorted(df["Gender"].dropna().unique())
    for dim in DIMENSION_ORDER:
        score = f"{dim}_score"
        for group, sub in df.groupby("Gender"):
            values = sub[score].dropna()
            low, high = ci95(values)
            summary_rows.append(
                {
                    "dimension": dim,
                    "Gender": group,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        if len(groups) == 2:
            a_name, b_name = groups
            a = df.loc[df["Gender"] == a_name, score].dropna()
            b = df.loc[df["Gender"] == b_name, score].dropna()
            t_res = stats.ttest_ind(a, b, equal_var=False)
            u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "dimension": dim,
                    "group_a": a_name,
                    "group_b": b_name,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_difference_a_minus_b": float(a.mean() - b.mean()),
                    "welch_t": float(t_res.statistic),
                    "welch_p": float(t_res.pvalue),
                    "mann_whitney_u": float(u_res.statistic),
                    "mann_whitney_p": float(u_res.pvalue),
                    "cohen_d_a_minus_b": cohen_d_independent(a, b),
                }
            )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["welch_p_holm"] = multipletests(tests["welch_p"], method="holm")[1]
        tests["mann_whitney_p_holm"] = multipletests(tests["mann_whitney_p"], method="holm")[1]
    return pd.DataFrame(summary_rows), tests


def continent_differences(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    omnibus_rows = []
    pairwise_rows = []
    continents = sorted(df["Continent"].dropna().unique())
    for dim in DIMENSION_ORDER:
        score = f"{dim}_score"
        for continent, sub in df.groupby("Continent"):
            values = sub[score].dropna()
            low, high = ci95(values)
            summary_rows.append(
                {
                    "dimension": dim,
                    "Continent": continent,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        arrays = [df.loc[df["Continent"] == c, score].dropna().to_numpy(dtype=float) for c in continents]
        f_res = stats.f_oneway(*arrays)
        h_res = stats.kruskal(*arrays)
        omnibus_rows.append(
            {
                "dimension": dim,
                "n_groups": len(continents),
                "anova_f": float(f_res.statistic),
                "anova_p": float(f_res.pvalue),
                "eta_squared": eta_squared_oneway(arrays),
                "kruskal_h": float(h_res.statistic),
                "kruskal_p": float(h_res.pvalue),
            }
        )
        for a_name, b_name in itertools.combinations(continents, 2):
            a = df.loc[df["Continent"] == a_name, score].dropna()
            b = df.loc[df["Continent"] == b_name, score].dropna()
            u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
            pairwise_rows.append(
                {
                    "dimension": dim,
                    "continent_a": a_name,
                    "continent_b": b_name,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_difference_a_minus_b": float(a.mean() - b.mean()),
                    "mann_whitney_u": float(u_res.statistic),
                    "p_value": float(u_res.pvalue),
                    "cohen_d_a_minus_b": cohen_d_independent(a, b),
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["anova_p_holm"] = multipletests(omnibus["anova_p"], method="holm")[1]
    omnibus["kruskal_p_holm"] = multipletests(omnibus["kruskal_p"], method="holm")[1]
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["p_holm_within_dimension"] = np.nan
    for dim in DIMENSION_ORDER:
        mask = pairwise["dimension"] == dim
        pairwise.loc[mask, "p_holm_within_dimension"] = multipletests(
            pairwise.loc[mask, "p_value"], method="holm"
        )[1]
    return pd.DataFrame(summary_rows), omnibus, pairwise


def correlation_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    score_cols = [f"{d}_score" for d in DIMENSION_ORDER]
    labels = DIMENSION_ORDER
    data = df[score_cols].rename(columns=dict(zip(score_cols, labels)))
    pearson_r = data.corr(method="pearson")
    spearman_r = data.corr(method="spearman")

    def p_matrix(method: str) -> pd.DataFrame:
        mat = pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels, dtype=float)
        for a, b in itertools.combinations(labels, 2):
            pair = data[[a, b]].dropna()
            if method == "pearson":
                _, p = stats.pearsonr(pair[a], pair[b])
            else:
                _, p = stats.spearmanr(pair[a], pair[b])
            mat.loc[a, b] = mat.loc[b, a] = p
        return mat

    return {
        "pearson_r": pearson_r,
        "pearson_p": p_matrix("pearson"),
        "spearman_rho": spearman_r,
        "spearman_p": p_matrix("spearman"),
    }


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return series * np.nan
    return (series - series.mean()) / sd


def path_coefficients_core(df: pd.DataFrame) -> dict[str, object]:
    model_df = df[[f"{d}_score" for d in DIMENSION_ORDER]].dropna().copy()
    model_df.columns = DIMENSION_ORDER
    for dim in DIMENSION_ORDER:
        model_df[f"{dim}_z"] = zscore(model_df[dim])
    formulas = {
        "M1_Resonance": "Resonance_z ~ Attraction_z",
        "M2_Retention": "Retention_z ~ Attraction_z + Resonance_z",
        "M3_Conversion": "Conversion_z ~ Attraction_z + Resonance_z + Retention_z",
    }
    models = {name: smf.ols(formula, data=model_df).fit() for name, formula in formulas.items()}
    coef_rows = []
    for name, model in models.items():
        outcome = formulas[name].split("~")[0].strip().replace("_z", "")
        for term in model.params.index:
            if term == "Intercept":
                continue
            coef_rows.append(
                {
                    "model": name,
                    "outcome": outcome,
                    "predictor": term.replace("_z", ""),
                    "standardized_beta": float(model.params[term]),
                    "std_error": float(model.bse[term]),
                    "t": float(model.tvalues[term]),
                    "p_value": float(model.pvalues[term]),
                    "ci95_low": float(model.conf_int().loc[term, 0]),
                    "ci95_high": float(model.conf_int().loc[term, 1]),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "n": int(model.nobs),
                }
            )
    summary_rows = []
    for name, model in models.items():
        summary_rows.append(
            {
                "model": name,
                "formula": formulas[name],
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
                "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "aic": float(model.aic),
                "bic": float(model.bic),
            }
        )
    indirect = path_indirect_effects_from_models(models)
    indirect_ci = bootstrap_indirect_effects(model_df, n_boot=5000, seed=20260526)
    indirect = indirect.merge(indirect_ci, on="effect", how="left")
    return {
        "model_df": model_df,
        "models": models,
        "coefficients": pd.DataFrame(coef_rows),
        "model_summary": pd.DataFrame(summary_rows),
        "indirect_effects": indirect,
    }


def path_indirect_effects_from_models(models: dict[str, object]) -> pd.DataFrame:
    a_ar = models["M1_Resonance"].params["Attraction_z"]
    b_at = models["M2_Retention"].params["Attraction_z"]
    b_rt = models["M2_Retention"].params["Resonance_z"]
    c_ac = models["M3_Conversion"].params["Attraction_z"]
    c_rc = models["M3_Conversion"].params["Resonance_z"]
    c_tc = models["M3_Conversion"].params["Retention_z"]
    rows = [
        ("Attraction -> Conversion direct", c_ac),
        ("Attraction -> Resonance -> Conversion", a_ar * c_rc),
        ("Attraction -> Retention -> Conversion", b_at * c_tc),
        ("Attraction -> Resonance -> Retention -> Conversion", a_ar * b_rt * c_tc),
        ("Attraction -> Conversion total indirect", a_ar * c_rc + b_at * c_tc + a_ar * b_rt * c_tc),
        ("Attraction -> Conversion total", c_ac + a_ar * c_rc + b_at * c_tc + a_ar * b_rt * c_tc),
        ("Resonance -> Retention -> Conversion", b_rt * c_tc),
    ]
    return pd.DataFrame(rows, columns=["effect", "estimate"])


def core_path_point_estimates(model_df: pd.DataFrame) -> dict[str, float]:
    df = model_df[DIMENSION_ORDER].copy()
    for dim in DIMENSION_ORDER:
        df[f"{dim}_z"] = zscore(df[dim])
    m1 = smf.ols("Resonance_z ~ Attraction_z", data=df).fit()
    m2 = smf.ols("Retention_z ~ Attraction_z + Resonance_z", data=df).fit()
    m3 = smf.ols("Conversion_z ~ Attraction_z + Resonance_z + Retention_z", data=df).fit()
    a_ar = m1.params["Attraction_z"]
    b_at = m2.params["Attraction_z"]
    b_rt = m2.params["Resonance_z"]
    c_ac = m3.params["Attraction_z"]
    c_rc = m3.params["Resonance_z"]
    c_tc = m3.params["Retention_z"]
    return {
        "Attraction -> Conversion direct": c_ac,
        "Attraction -> Resonance -> Conversion": a_ar * c_rc,
        "Attraction -> Retention -> Conversion": b_at * c_tc,
        "Attraction -> Resonance -> Retention -> Conversion": a_ar * b_rt * c_tc,
        "Attraction -> Conversion total indirect": a_ar * c_rc + b_at * c_tc + a_ar * b_rt * c_tc,
        "Attraction -> Conversion total": c_ac + a_ar * c_rc + b_at * c_tc + a_ar * b_rt * c_tc,
        "Resonance -> Retention -> Conversion": b_rt * c_tc,
    }


def bootstrap_indirect_effects(model_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {}
    n = len(model_df)
    base = model_df[DIMENSION_ORDER].reset_index(drop=True)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = base.iloc[idx].copy()
        vals = core_path_point_estimates(sample)
        for key, value in vals.items():
            estimates.setdefault(key, []).append(float(value))
    rows = []
    for effect, values in estimates.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "effect": effect,
                "bootstrap_ci95_low": float(np.percentile(arr, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(arr, 97.5)),
                "bootstrap_se": float(np.std(arr, ddof=1)),
                "bootstrap_n": n_boot,
            }
        )
    return pd.DataFrame(rows)


def controlled_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df[["Gender", "Continent"] + [f"{d}_score" for d in DIMENSION_ORDER]].dropna().copy()
    for dim in DIMENSION_ORDER:
        model_df[f"{dim}_z"] = zscore(model_df[f"{dim}_score"])
    formulas = {
        "M1_Resonance_controlled": "Resonance_z ~ Attraction_z + C(Gender) + C(Continent)",
        "M2_Retention_controlled": (
            "Retention_z ~ Attraction_z + Resonance_z + C(Gender) + C(Continent)"
        ),
        "M3_Conversion_controlled": (
            "Conversion_z ~ Attraction_z + Resonance_z + Retention_z + C(Gender) + C(Continent)"
        ),
    }
    rows = []
    summaries = []
    for name, formula in formulas.items():
        model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
        conf = model.conf_int()
        summaries.append(
            {
                "model": name,
                "formula": formula,
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
                "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "covariance": "HC3 robust SE",
            }
        )
        for term in model.params.index:
            rows.append(
                {
                    "model": name,
                    "term": term,
                    "estimate": float(model.params[term]),
                    "std_error": float(model.bse[term]),
                    "t": float(model.tvalues[term]),
                    "p_value": float(model.pvalues[term]),
                    "ci95_low": float(conf.loc[term, 0]),
                    "ci95_high": float(conf.loc[term, 1]),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def first_round_overview() -> pd.DataFrame:
    if not FIRST_ROUND_XLSX.exists():
        return pd.DataFrame([{"note": "First-round workbook not found; no overview generated."}])
    df = pd.read_excel(FIRST_ROUND_XLSX)
    rows = [
        {"metric": "data_rows", "value": df.shape[0]},
        {"metric": "columns", "value": df.shape[1]},
        {
            "metric": "integration_decision",
            "value": (
                "Not merged into second-round scale analysis: the first-round workbook is a "
                "different questionnaire with a small pilot sample and non-matching item structure."
            ),
        },
    ]
    if df.shape[1] >= 8:
        gender_col = df.columns[7]
        for value, count in df[gender_col].value_counts(dropna=False).items():
            rows.append({"metric": f"first_round_gender_code_{value}", "value": int(count)})
    if df.shape[1] >= 9:
        country_col = df.columns[8]
        top_country = df[country_col].value_counts(dropna=False).head(10)
        for value, count in top_country.items():
            rows.append({"metric": f"first_round_country_{value}", "value": int(count)})
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=index, encoding="utf-8-sig")
    elif path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=index)
    else:
        raise ValueError(f"Unsupported output suffix: {path.suffix}")
    return path


def save_matrix_tables(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31])
    return path


def save_all_tables(table_map: dict[str, pd.DataFrame]) -> list[Path]:
    paths = []
    for name, table in table_map.items():
        paths.append(save_dataframe(table, TABLES_DIR / f"{name}.csv"))
    xlsx_path = TABLES_DIR / "paper_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, table in table_map.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    paths.append(xlsx_path)
    return paths


def figure_sample_profile(profile: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, var in zip(axes, ["Gender", "Continent"]):
        sub = profile[profile["variable"] == var].sort_values("n", ascending=False)
        ax.bar(sub["category"], sub["n"], color="#4C78A8")
        ax.set_title(var)
        ax.set_ylabel("n")
        ax.tick_params(axis="x", rotation=30)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, row["n"] + 1, f"{int(row['n'])}", ha="center", va="bottom", fontsize=9)
    return save_figure(fig, "fig01_sample_profile")


def figure_dimension_means(desc: pd.DataFrame) -> list[Path]:
    sub = desc[desc["dimension"].isin(DIMENSION_ORDER)].set_index("dimension").loc[DIMENSION_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = np.arange(len(sub))
    y = sub["mean"].to_numpy()
    yerr = np.vstack([y - sub["ci95_low"].to_numpy(), sub["ci95_high"].to_numpy() - y])
    ax.bar(x, y, yerr=yerr, capsize=5, color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"])
    ax.set_xticks(x, sub["dimension"])
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean score on five-point Likert scale")
    ax.set_title("Dimension Scores with 95% CI")
    for i, value in enumerate(y):
        ax.text(i, value + 0.06, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    return save_figure(fig, "fig02_dimension_means_ci")


def figure_dimension_boxplots(df: pd.DataFrame) -> list[Path]:
    values = [df[f"{dim}_score"].dropna().to_numpy() for dim in DIMENSION_ORDER]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.boxplot(values, tick_labels=DIMENSION_ORDER, patch_artist=True)
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Five-point Likert score")
    ax.set_title("Distribution of Dimension Scores")
    return save_figure(fig, "fig03_dimension_boxplots")


def figure_correlation_heatmap(corr: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    data = corr.loc[DIMENSION_ORDER, DIMENSION_ORDER].to_numpy()
    im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(DIMENSION_ORDER)), DIMENSION_ORDER, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(DIMENSION_ORDER)), DIMENSION_ORDER)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pearson Correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, "fig04_correlation_heatmap")


def figure_group_means(gender_summary: pd.DataFrame, continent_summary: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    pivot_gender = gender_summary.pivot(index="dimension", columns="Gender", values="mean").loc[DIMENSION_ORDER]
    x = np.arange(len(DIMENSION_ORDER))
    width = 0.35
    for i, group in enumerate(pivot_gender.columns):
        offset = (i - (len(pivot_gender.columns) - 1) / 2) * width
        axes[0].bar(x + offset, pivot_gender[group], width=width, label=group)
    axes[0].set_xticks(x, DIMENSION_ORDER, rotation=20)
    axes[0].set_ylim(1, 5)
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Dimension Means by Gender")
    axes[0].legend(frameon=False)

    pivot_continent = continent_summary.pivot(index="dimension", columns="Continent", values="mean").loc[DIMENSION_ORDER]
    for continent in pivot_continent.columns:
        axes[1].plot(DIMENSION_ORDER, pivot_continent[continent], marker="o", label=continent)
    axes[1].set_ylim(1, 5)
    axes[1].set_ylabel("Mean score")
    axes[1].set_title("Dimension Means by Continent")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=8)
    return save_figure(fig, "fig05_group_means")


def figure_path_model(path_coef: pd.DataFrame) -> list[Path]:
    coef_lookup = {
        (row["predictor"], row["outcome"]): row["standardized_beta"]
        for _, row in path_coef.iterrows()
        if row["predictor"] in DIMENSION_ORDER and row["outcome"] in DIMENSION_ORDER
    }
    positions = {
        "Attraction": (0.10, 0.52),
        "Resonance": (0.37, 0.52),
        "Retention": (0.64, 0.52),
        "Conversion": (0.91, 0.52),
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.98)
    ax.axis("off")
    for dim, (x, y) in positions.items():
        ax.text(
            x,
            y,
            dim,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#555555"),
        )

    def arrow(start: str, end: str, label_xy: tuple[float, float], curve: float = 0.0) -> None:
        beta = coef_lookup.get((start, end), np.nan)
        color = "#2F5597" if beta >= 0 else "#C00000"
        width = 1.4 + 4 * min(abs(beta), 0.8) if not np.isnan(beta) else 1.0
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=16,
            lw=width,
            color=color,
            shrinkA=35,
            shrinkB=38,
            connectionstyle=f"arc3,rad={curve}",
        )
        ax.add_patch(patch)
        ax.text(
            label_xy[0],
            label_xy[1],
            f"β={beta:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )

    arrow("Attraction", "Resonance", (0.235, 0.62), 0.0)
    arrow("Resonance", "Retention", (0.505, 0.62), 0.0)
    arrow("Retention", "Conversion", (0.775, 0.62), 0.0)
    arrow("Attraction", "Retention", (0.37, 0.26), -0.34)
    arrow("Resonance", "Conversion", (0.64, 0.82), 0.34)
    arrow("Attraction", "Conversion", (0.50, 0.08), -0.46)
    ax.set_title("Core Standardized Path Model")
    return save_figure(fig, "fig06_path_model")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for suffix in [".png", ".pdf"]:
        path = FIGURES_DIR / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_summary_markdown(
    path: Path,
    cleaned: pd.DataFrame,
    desc: pd.DataFrame,
    alpha: pd.DataFrame,
    dim_overall: pd.DataFrame,
    dim_pairwise: pd.DataFrame,
    gender_tests: pd.DataFrame,
    continent_omnibus: pd.DataFrame,
    corr: dict[str, pd.DataFrame],
    path_summary: pd.DataFrame,
    path_coef: pd.DataFrame,
    indirect: pd.DataFrame,
) -> Path:
    ddesc = desc[desc["dimension"].isin(DIMENSION_ORDER)].set_index("dimension")
    alpha_s = alpha.set_index("construct")["cronbach_alpha"]
    pearson = corr["pearson_r"]

    def fmt_p(p: float) -> str:
        if pd.isna(p):
            return "NA"
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    strongest_corr = (
        pearson.where(~np.eye(len(pearson), dtype=bool))
        .stack()
        .rename("r")
        .reset_index()
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values("abs_r", ascending=False)
        .iloc[0]
    )
    dim_chi = dim_overall.iloc[0]
    sig_gender = gender_tests[gender_tests["mann_whitney_p_holm"] < 0.05]
    sig_continent = continent_omnibus[continent_omnibus["kruskal_p_holm"] < 0.05]
    conv_model = path_summary[path_summary["model"] == "M3_Conversion"].iloc[0]
    conv_terms = path_coef[path_coef["model"] == "M3_Conversion"].copy()
    conv_terms = conv_terms.sort_values("standardized_beta", ascending=False)
    total_indirect = indirect[indirect["effect"] == "Attraction -> Conversion total indirect"].iloc[0]

    lines = [
        "# Quantitative Analysis Summary",
        "",
        f"- Data: second-round survey, N = {len(cleaned)} valid rows; no original file was overwritten.",
        f"- Scale direction: {SCORE_NOTE}",
        "- Dimension means on the five-point Likert scale: "
        + "; ".join(f"{dim} = {ddesc.loc[dim, 'mean']:.3f}" for dim in DIMENSION_ORDER)
        + ".",
        "- Reliability Cronbach alpha: "
        + "; ".join(f"{dim} = {alpha_s.loc[dim]:.3f}" for dim in DIMENSION_ORDER)
        + f"; overall 19-item alpha = {alpha_s.loc['Overall_19_items']:.3f}.",
        (
            f"- Dimension difference: Friedman chi-square({int(dim_chi['df'])}) = "
            f"{dim_chi['statistic_chi2']:.3f}, p {fmt_p(dim_chi['p_value'])}, "
            f"Kendall's W = {dim_chi['effect_kendalls_w']:.3f}."
        ),
        "- Largest within-person dimension gaps: "
        + "; ".join(
            f"{row.dimension_a} - {row.dimension_b} = {row.mean_difference_a_minus_b:.3f}, Holm p {fmt_p(row.p_holm)}"
            for row in dim_pairwise.reindex(dim_pairwise["mean_difference_a_minus_b"].abs().sort_values(ascending=False).index).head(3).itertuples()
        )
        + ".",
        (
            "- Gender differences after Holm correction: "
            + ("none significant." if sig_gender.empty else "; ".join(sig_gender["dimension"].tolist()) + ".")
        ),
        (
            "- Continent differences after Holm correction: "
            + ("none significant." if sig_continent.empty else "; ".join(sig_continent["dimension"].tolist()) + ".")
        ),
        (
            f"- Strongest dimension correlation: {strongest_corr['level_0']} with {strongest_corr['level_1']}, "
            f"Pearson r = {strongest_corr['r']:.3f}."
        ),
        (
            f"- Core path model for Conversion: R2 = {conv_model['r_squared']:.3f}; predictors "
            + "; ".join(f"{row.predictor} beta = {row.standardized_beta:.3f}, p {fmt_p(row.p_value)}" for row in conv_terms.itertuples())
            + "."
        ),
        (
            f"- Total indirect Attraction-to-Conversion effect through Resonance/Retention = "
            f"{total_indirect['estimate']:.3f}, bootstrap 95% CI "
            f"[{total_indirect['bootstrap_ci95_low']:.3f}, {total_indirect['bootstrap_ci95_high']:.3f}]."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> SavedOutputs:
    ensure_directories()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    raw = read_second_round()
    cleaned, validation = validate_and_score(raw)

    codebook = make_codebook()
    profile = sample_profile(cleaned)
    item_desc = item_descriptives(cleaned)
    dim_desc = dimension_descriptives(cleaned)
    alpha, alpha_items = reliability_tables(cleaned)
    dim_overall, dim_pairwise = dimension_difference_tests(cleaned)
    gender_summary, gender_tests = gender_differences(cleaned)
    continent_summary, continent_omnibus, continent_pairwise = continent_differences(cleaned)
    corr_tables = correlation_tables(cleaned)
    path_core = path_coefficients_core(cleaned)
    controlled_summary, controlled_coefficients = controlled_regressions(cleaned)
    first_overview = first_round_overview()

    data_paths = [
        save_dataframe(cleaned, OUTPUTS_DIR / "second_round_cleaned_with_scores.csv"),
        save_dataframe(cleaned, OUTPUTS_DIR / "second_round_cleaned_with_scores.xlsx"),
        save_dataframe(
            cleaned.melt(
                id_vars=["ID", "Gender", "Continent"],
                value_vars=[f"{dim}_score" for dim in DIMENSION_ORDER],
                var_name="dimension",
                value_name="score",
            ).assign(dimension=lambda x: x["dimension"].str.replace("_score", "", regex=False)),
            OUTPUTS_DIR / "dimension_scores_long.csv",
        ),
        save_dataframe(validation, OUTPUTS_DIR / "data_validation_checks.csv"),
        save_dataframe(first_overview, OUTPUTS_DIR / "first_round_overview.csv"),
    ]

    table_map = {
        "01_codebook": codebook,
        "02_sample_profile": profile,
        "03_item_descriptives": item_desc,
        "04_dimension_descriptives": dim_desc,
        "05_cronbach_alpha": alpha,
        "05b_item_reliability": alpha_items,
        "06_dimension_difference_overall": dim_overall,
        "06b_dimension_pairwise": dim_pairwise,
        "07_gender_summary": gender_summary,
        "07b_gender_tests": gender_tests,
        "08_continent_summary": continent_summary,
        "08b_continent_omnibus": continent_omnibus,
        "08c_continent_pairwise": continent_pairwise,
        "09_pearson_r": corr_tables["pearson_r"].reset_index(names="dimension"),
        "09b_pearson_p": corr_tables["pearson_p"].reset_index(names="dimension"),
        "09c_spearman_rho": corr_tables["spearman_rho"].reset_index(names="dimension"),
        "09d_spearman_p": corr_tables["spearman_p"].reset_index(names="dimension"),
        "10_path_model_summary": path_core["model_summary"],
        "10b_path_coefficients": path_core["coefficients"],
        "10c_indirect_effects": path_core["indirect_effects"],
        "10d_controlled_regression_summary": controlled_summary,
        "10e_controlled_regression_coefficients": controlled_coefficients,
        "11_first_round_overview": first_overview,
        "12_data_validation_checks": validation,
    }
    table_paths = save_all_tables(table_map)
    table_paths.append(save_matrix_tables(corr_tables, TABLES_DIR / "correlation_matrices.xlsx"))

    figure_paths: list[Path] = []
    figure_paths += figure_sample_profile(profile)
    figure_paths += figure_dimension_means(dim_desc)
    figure_paths += figure_dimension_boxplots(cleaned)
    figure_paths += figure_correlation_heatmap(corr_tables["pearson_r"])
    figure_paths += figure_group_means(gender_summary, continent_summary)
    figure_paths += figure_path_model(path_core["coefficients"])

    summary_path = write_summary_markdown(
        OUTPUTS_DIR / "analysis_summary.md",
        cleaned,
        dim_desc,
        alpha,
        dim_overall,
        dim_pairwise,
        gender_tests,
        continent_omnibus,
        corr_tables,
        path_core["model_summary"],
        path_core["coefficients"],
        path_core["indirect_effects"],
    )
    data_paths.append(summary_path)

    manifest_rows = []
    for category, paths in [("table", table_paths), ("figure", figure_paths), ("data", data_paths)]:
        for path in paths:
            manifest_rows.append({"category": category, "path": str(path.relative_to(PROJECT_ROOT))})
    manifest = pd.DataFrame(manifest_rows)
    data_paths.append(save_dataframe(manifest, OUTPUTS_DIR / "output_manifest.csv"))

    print("Analysis complete.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"N: {len(cleaned)}")
    print("Dimension means:")
    for row in dim_desc[dim_desc["dimension"].isin(DIMENSION_ORDER)].itertuples():
        print(f"  {row.dimension}: mean={row.mean:.3f}, sd={row.sd:.3f}")
    print("Cronbach alpha:")
    for row in alpha.itertuples():
        print(f"  {row.construct}: alpha={row.cronbach_alpha:.3f}")
    return SavedOutputs(tables=table_paths, figures=figure_paths, data=data_paths)


if __name__ == "__main__":
    main()
