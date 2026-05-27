#!/usr/bin/env python3
"""Supplementary factorability and EFA checks for the second-round survey.

The script reads the cleaned second-round analysis output and writes derived
tables only. It does not modify the source workbooks or cleaned data files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "analysis" / "outputs" / "second_round_cleaned_with_scores.csv"
TABLES = PROJECT / "analysis" / "tables"

DIMENSIONS = {
    "Attraction": [f"Attraction_{i}" for i in range(1, 5)],
    "Resonance": [f"Resonance_{i}" for i in range(1, 6)],
    "Retention": [f"Retention_{i}" for i in range(1, 6)],
    "Conversion": [f"Conversion_{i}" for i in range(1, 6)],
}
DIMENSION_ORDER = list(DIMENSIONS)
ITEM_COLUMNS = [item for items in DIMENSIONS.values() for item in items]


def save_csv(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def kmo_from_corr(corr: np.ndarray) -> tuple[float, np.ndarray]:
    inv_corr = np.linalg.pinv(corr)
    scale = np.diag(1 / np.sqrt(np.diag(inv_corr)))
    partial = -scale @ inv_corr @ scale
    np.fill_diagonal(partial, 0)

    corr_sq = corr.copy()
    np.fill_diagonal(corr_sq, 0)
    corr_sq = corr_sq**2
    partial_sq = partial**2

    item_kmo = corr_sq.sum(axis=0) / (corr_sq.sum(axis=0) + partial_sq.sum(axis=0))
    overall_kmo = corr_sq.sum() / (corr_sq.sum() + partial_sq.sum())
    return float(overall_kmo), item_kmo


def bartlett_from_corr(corr: np.ndarray, n: int) -> tuple[float, int, float, float]:
    p = corr.shape[0]
    det = float(np.linalg.det(corr))
    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) // 2
    p_value = float(stats.chi2.sf(chi_square, df))
    return float(chi_square), int(df), p_value, det


def initial_communalities(corr: np.ndarray) -> np.ndarray:
    inv_corr = np.linalg.pinv(corr)
    smc = 1 - 1 / np.diag(inv_corr)
    return np.clip(smc, 0.05, 0.99)


def principal_axis_factoring(
    corr: np.ndarray,
    n_factors: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    reduced = corr.copy()
    communalities = initial_communalities(corr)
    for iteration in range(1, max_iter + 1):
        np.fill_diagonal(reduced, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        retained = np.maximum(eigenvalues[:n_factors], 0)
        loadings = eigenvectors[:, :n_factors] * np.sqrt(retained)
        updated = np.clip(np.sum(loadings**2, axis=1), 0, 0.99)
        if np.max(np.abs(updated - communalities)) < tol:
            return loadings, eigenvalues, updated, iteration
        communalities = updated
    return loadings, eigenvalues, communalities, max_iter


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n_rows, n_cols = loadings.shape
    rotation = np.eye(n_cols)
    last_objective = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, singular_values, vh = np.linalg.svd(
            loadings.T
            @ (rotated**3 - (gamma / n_rows) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        objective = singular_values.sum()
        if last_objective and objective / last_objective < 1 + tol:
            break
        last_objective = objective
    return loadings @ rotation


def align_factors(loadings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    construct_scores = np.zeros((len(DIMENSION_ORDER), loadings.shape[1]))
    item_index = {item: idx for idx, item in enumerate(ITEM_COLUMNS)}
    for i, construct in enumerate(DIMENSION_ORDER):
        indices = [item_index[item] for item in DIMENSIONS[construct]]
        construct_scores[i, :] = np.abs(loadings[indices, :]).mean(axis=0)

    construct_assignment, factor_assignment = linear_sum_assignment(-construct_scores)
    mapping = dict(zip(construct_assignment, factor_assignment))

    aligned_columns = []
    rows = []
    for construct_index, construct in enumerate(DIMENSION_ORDER):
        factor_index = mapping[construct_index]
        indices = [item_index[item] for item in DIMENSIONS[construct]]
        column = loadings[:, factor_index].copy()
        if column[indices].mean() < 0:
            column *= -1
        aligned_columns.append(column)
        rows.append(
            {
                "construct": construct,
                "source_factor_index": int(factor_index + 1),
                "mean_abs_loading_on_construct_items": float(construct_scores[construct_index, factor_index]),
            }
        )

    aligned = pd.DataFrame(
        np.column_stack(aligned_columns),
        index=ITEM_COLUMNS,
        columns=[f"{construct}_factor" for construct in DIMENSION_ORDER],
    )
    return aligned, pd.DataFrame(rows)


def parallel_analysis(data: pd.DataFrame, n_iter: int = 1000, seed: int = 20260527) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual = np.linalg.eigvalsh(data.corr().to_numpy())[::-1]
    simulated = np.empty((n_iter, data.shape[1]))
    base = data.to_numpy(dtype=float)
    for i in range(n_iter):
        permuted = np.column_stack([rng.permutation(base[:, j]) for j in range(base.shape[1])])
        simulated[i, :] = np.linalg.eigvalsh(np.corrcoef(permuted, rowvar=False))[::-1]
    return pd.DataFrame(
        {
            "factor_number": np.arange(1, data.shape[1] + 1),
            "actual_eigenvalue": actual,
            "permutation_mean_eigenvalue": simulated.mean(axis=0),
            "permutation_p95_eigenvalue": np.percentile(simulated, 95, axis=0),
            "retain_by_parallel_p95": actual > np.percentile(simulated, 95, axis=0),
            "retain_by_kaiser_gt1": actual > 1,
        }
    )


def adjusted_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(DIMENSION_ORDER):
        for right in DIMENSION_ORDER[i + 1 :]:
            pair = df[[f"{left}_score", f"{right}_score"]].dropna()
            pearson_r, pearson_p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            spearman_r, spearman_p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append(
                {
                    "dimension_a": left,
                    "dimension_b": right,
                    "n": int(len(pair)),
                    "pearson_r": float(pearson_r),
                    "pearson_p_raw": float(pearson_p),
                    "spearman_rho": float(spearman_r),
                    "spearman_p_raw": float(spearman_p),
                }
            )
    out = pd.DataFrame(rows)
    out["pearson_p_holm"] = multipletests(out["pearson_p_raw"], method="holm")[1]
    out["pearson_p_fdr_bh"] = multipletests(out["pearson_p_raw"], method="fdr_bh")[1]
    out["spearman_p_holm"] = multipletests(out["spearman_p_raw"], method="holm")[1]
    out["spearman_p_fdr_bh"] = multipletests(out["spearman_p_raw"], method="fdr_bh")[1]
    return out


def main() -> None:
    df = pd.read_csv(DATA)
    item_data = df[ITEM_COLUMNS].dropna().astype(float)
    n, n_items = item_data.shape
    corr = item_data.corr().to_numpy()

    kmo_overall, item_kmo = kmo_from_corr(corr)
    bartlett_chi2, bartlett_df, bartlett_p, determinant = bartlett_from_corr(corr, n)
    summary = pd.DataFrame(
        [
            {
                "n_complete": n,
                "n_items": n_items,
                "correlation_type": "Pearson item correlation",
                "kmo_overall": kmo_overall,
                "bartlett_chi_square": bartlett_chi2,
                "bartlett_df": bartlett_df,
                "bartlett_p_value": bartlett_p,
                "correlation_determinant": determinant,
                "efa_extraction": "principal axis factoring",
                "efa_rotation": "orthogonal varimax",
            }
        ]
    )
    item_kmo_table = pd.DataFrame(
        {
            "item": ITEM_COLUMNS,
            "intended_construct": [item.split("_")[0] for item in ITEM_COLUMNS],
            "kmo": item_kmo,
        }
    )

    eigen = parallel_analysis(item_data)
    retained_factors = int(eigen["retain_by_parallel_p95"].sum())
    n_factors = 4 if retained_factors == 4 else retained_factors
    n_factors = max(1, min(n_factors, len(DIMENSION_ORDER)))

    unrotated, _, communalities, iterations = principal_axis_factoring(corr, n_factors=n_factors)
    rotated = varimax(unrotated)
    aligned, alignment = align_factors(rotated)
    aligned["item"] = aligned.index
    aligned["intended_construct"] = [item.split("_")[0] for item in aligned.index]
    factor_cols = [f"{construct}_factor" for construct in DIMENSION_ORDER]
    aligned["communality"] = np.sum(aligned[factor_cols].to_numpy() ** 2, axis=1)
    aligned["uniqueness"] = 1 - aligned["communality"]
    aligned["primary_factor"] = aligned[factor_cols].abs().idxmax(axis=1).str.replace("_factor", "", regex=False)
    aligned["primary_loading"] = [
        aligned.loc[item, f"{factor}_factor"] for item, factor in zip(aligned.index, aligned["primary_factor"])
    ]
    loading_table = aligned[["item", "intended_construct", *factor_cols, "primary_factor", "primary_loading", "communality", "uniqueness"]]

    variance = pd.DataFrame(
        {
            "factor": [col.replace("_factor", "") for col in factor_cols],
            "ss_loadings": np.sum(aligned[factor_cols].to_numpy() ** 2, axis=0),
        }
    )
    variance["proportion_total_variance"] = variance["ss_loadings"] / n_items
    variance["cumulative_total_variance"] = variance["proportion_total_variance"].cumsum()
    variance["paf_iterations"] = iterations

    save_csv(summary, "13_efa_kmo_bartlett.csv")
    save_csv(item_kmo_table, "13a_efa_item_kmo.csv")
    save_csv(eigen, "13b_efa_eigenvalues_parallel.csv")
    save_csv(loading_table, "13c_efa_varimax_loadings.csv")
    save_csv(variance, "13d_efa_factor_variance.csv")
    save_csv(alignment, "13e_efa_factor_alignment.csv")
    save_csv(adjusted_dimension_correlations(df), "09e_dimension_correlations_adjusted.csv")

    print(f"Saved EFA diagnostics for n={n}, items={n_items}, factors={n_factors}.")
    print(f"KMO={kmo_overall:.3f}; Bartlett chi-square({bartlett_df})={bartlett_chi2:.3f}, p={bartlett_p:.3g}.")


if __name__ == "__main__":
    main()
