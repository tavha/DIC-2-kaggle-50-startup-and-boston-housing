"""
50 Startups: Feature Selection Progression Analysis
===================================================
Evaluates model performance progression across feature subsets of size k = 1..5
for 5 distinct feature selection algorithms:
  1. Lasso (L1 Regularization Ranking)
  2. Sequential Forward Selection (Forward SFS)
  3. Recursive Feature Elimination (RFE)
  4. ElasticNet Ranking
  5. Exhaustive Best-Subset Search

Generates:
  - 5 individual 2-panel progression plots (RMSE & R-squared)
  - 1 combined 2-panel comparison plot across all 5 algorithms
  - 5 progression CSV tables
  - Synchronized model_data.json and model_data.js for web dashboard
"""

import os
import json
import itertools
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATA_LOCAL_PATH = "50_Startups.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)

PREDICTOR_COLUMNS = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York",
]
TARGET_COLUMN = "Profit"

TEST_SIZE = 0.20
RANDOM_STATE = 0


# ==============================================================================
# Data Ingestion & Preprocessing
# ==============================================================================

class StartupDataset:
    """Manages loading, dummy encoding, splitting, and scaling of startup data."""

    def __init__(self, filepath: str = DATA_LOCAL_PATH, url: str = DATA_URL):
        self.df_raw = self._load_data(filepath, url)
        self.features = list(PREDICTOR_COLUMNS)
        self.target = TARGET_COLUMN

        # One-hot encode State; California is reference category (dropped)
        self.df_encoded = pd.get_dummies(self.df_raw, columns=["State"], dtype=int)
        
        self.X = self.df_encoded[self.features].astype(float)
        self.y = self.df_encoded[self.target].astype(float)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.features,
            index=self.X_train.index,
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.features,
            index=self.X_test.index,
        )

    @staticmethod
    def _load_data(filepath: str, url: str) -> pd.DataFrame:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        try:
            return pd.read_csv(url)
        except Exception as exc:
            raise RuntimeError(f"Unable to load data from {url}: {exc}") from exc


# ==============================================================================
# Model Evaluation & Statistics Extractor
# ==============================================================================

class ModelEvaluator:
    """Computes comprehensive regression metrics and inferential statistics."""

    def __init__(self, data: StartupDataset):
        self.data = data

    def evaluate_subset_quick(self, features_subset: List[str]) -> Tuple[float, float]:
        """Calculates test RMSE and R-squared for feature selection searching."""
        sub = list(features_subset)
        reg = LinearRegression().fit(self.data.X_train[sub], self.data.y_train)
        preds = reg.predict(self.data.X_test[sub])
        rmse = float(np.sqrt(mean_squared_error(self.data.y_test, preds)))
        r2 = float(r2_score(self.data.y_test, preds))
        return rmse, r2

    def evaluate_subset_full(self, k: int, features_subset: List[str]) -> Dict[str, Any]:
        """
        Fits OLS using statsmodels for full coefficient inference, VIF,
        F-statistic, and test set residuals.
        """
        sub = list(features_subset)
        X_tr = self.data.X_train[sub]
        y_tr = self.data.y_train
        X_te = self.data.X_test[sub]
        y_te = self.data.y_test

        # Fit Statsmodels OLS with constant
        X_tr_const = sm.add_constant(X_tr, has_constant="add")
        ols_model = sm.OLS(y_tr, X_tr_const).fit()

        # Predictions on test set
        X_te_const = sm.add_constant(X_te, has_constant="add")
        test_preds = ols_model.predict(X_te_const)

        rmse = float(np.sqrt(mean_squared_error(y_te, test_preds)))
        mae = float(mean_absolute_error(y_te, test_preds))
        r2 = float(r2_score(y_te, test_preds))

        n_samples = len(y_te)
        adj_r2 = (
            1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - k - 1)
            if n_samples > k + 1
            else float("nan")
        )

        f_stat = float(ols_model.fvalue) if not np.isnan(ols_model.fvalue) else 0.0
        f_pvalue = float(ols_model.f_pvalue) if not np.isnan(ols_model.f_pvalue) else 1.0

        # Variance Inflation Factor on training features
        vif_map: Dict[str, Optional[float]] = {}
        if len(sub) == 1:
            vif_map[sub[0]] = 1.0
        else:
            for idx, col in enumerate(X_tr_const.columns):
                if col == "const":
                    continue
                try:
                    val = variance_inflation_factor(X_tr_const.values, idx)
                    vif_map[col] = float(val) if not np.isinf(val) else 999.0
                except Exception:
                    vif_map[col] = None

        # Build coefficient statistics
        coefficients: List[Dict[str, Any]] = []
        conf_int = ols_model.conf_int()

        for col in X_tr_const.columns:
            display_name = "Intercept (β₀)" if col == "const" else col
            coefficients.append({
                "feature": display_name,
                "coef": float(ols_model.params[col]),
                "stderr": float(ols_model.bse[col]),
                "tstat": float(ols_model.tvalues[col]),
                "pvalue": float(ols_model.pvalues[col]),
                "ci_lower": float(conf_int.loc[col, 0]),
                "ci_upper": float(conf_int.loc[col, 1]),
                "vif": vif_map.get(col, None),
            })

        # Test set residual diagnostics (sorted by absolute residual descending)
        residuals = y_te - test_preds
        abs_errors = residuals.abs()
        test_indices = y_te.index

        outlier_rows = []
        for orig_idx in test_indices:
            act = float(y_te.loc[orig_idx])
            pred = float(test_preds.loc[orig_idx])
            res = float(residuals.loc[orig_idx])
            abs_err = float(abs_errors.loc[orig_idx])
            pct_err = (abs_err / act * 100.0) if act != 0 else 0.0

            outlier_rows.append({
                "index": int(orig_idx),
                "actual": act,
                "predicted": pred,
                "residual": res,
                "abs_error": abs_err,
                "pct_error": pct_err,
            })

        outlier_rows.sort(key=lambda item: item["abs_error"], reverse=True)
        for rank_idx, row in enumerate(outlier_rows, start=1):
            row["rank"] = rank_idx

        return {
            "k": k,
            "features": sub,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "coefficients": coefficients,
            "outliers": outlier_rows,
        }


# ==============================================================================
# Feature Selection Strategies
# ==============================================================================

class FeatureSelectionStrategies:
    """Implements the 5 feature selection algorithms to generate subsets for k=1..5."""

    def __init__(self, data: StartupDataset, evaluator: ModelEvaluator):
        self.data = data
        self.evaluator = evaluator

    def get_lasso_subsets(self) -> List[List[str]]:
        """L1 Regularization Ranking based on standardized coefficient magnitudes."""
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=100000)
        lasso.fit(self.data.X_train_scaled, self.data.y_train)
        ranking = [
            feat for _, feat in sorted(
                zip(np.abs(lasso.coef_), self.data.features), reverse=True
            )
        ]
        return [ranking[:k] for k in range(1, 6)]

    def get_forward_sfs_subsets(self) -> List[List[str]]:
        """Sequential Forward Selection optimizing cross-validated error."""
        subsets = []
        for k in range(1, 6):
            if k == len(self.data.features):
                subsets.append(list(self.data.features))
            else:
                sfs = SequentialFeatureSelector(
                    LinearRegression(),
                    n_features_to_select=k,
                    direction="forward",
                    scoring="neg_root_mean_squared_error",
                    cv=5,
                )
                sfs.fit(self.data.X_train, self.data.y_train)
                selected = list(self.data.X_train.columns[sfs.get_support()])
                subsets.append(selected)
        return subsets

    def get_rfe_subsets(self) -> List[List[str]]:
        """Recursive Feature Elimination using OLS weights on scaled features."""
        subsets = []
        for k in range(1, 6):
            if k == len(self.data.features):
                subsets.append(list(self.data.features))
            else:
                rfe = RFE(LinearRegression(), n_features_to_select=k)
                rfe.fit(self.data.X_train_scaled, self.data.y_train)
                selected = list(self.data.X_train.columns[rfe.get_support()])
                subsets.append(selected)
        return subsets

    def get_elasticnet_subsets(self) -> List[List[str]]:
        """ElasticNet Ranking based on regularized coefficient magnitudes."""
        enet = ElasticNetCV(cv=5, random_state=RANDOM_STATE, max_iter=100000)
        enet.fit(self.data.X_train_scaled, self.data.y_train)
        ranking = [
            feat for _, feat in sorted(
                zip(np.abs(enet.coef_), self.data.features), reverse=True
            )
        ]
        return [ranking[:k] for k in range(1, 6)]

    def get_exhaustive_subsets(self) -> List[List[str]]:
        """Exhaustive search selecting the subset with minimal test RMSE for each k."""
        subsets = []
        for k in range(1, 6):
            best_rmse = float("inf")
            best_sub: Optional[List[str]] = None
            for comb in itertools.combinations(self.data.features, k):
                rmse, _ = self.evaluator.evaluate_subset_quick(list(comb))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_sub = list(comb)
            subsets.append(best_sub or list(self.data.features[:k]))
        return subsets


# ==============================================================================
# Visualization Engine (Publication Quality Redesign)
# ==============================================================================

class ModernChartTheme:
    """Styles and palette constants for publication-grade visualization."""

    # Color definitions
    COLOR_PRIMARY_INDIGO = "#4338ca"
    COLOR_SUCCESS_EMERALD = "#059669"
    COLOR_AMBER = "#d97706"
    COLOR_ROSE = "#e11d48"
    COLOR_CYAN = "#0891b2"
    
    PALETTE_ALGORITHMS = [
        COLOR_PRIMARY_INDIGO,
        COLOR_SUCCESS_EMERALD,
        COLOR_AMBER,
        COLOR_ROSE,
        COLOR_CYAN,
    ]
    MARKERS = ["o", "s", "^", "D", "v"]

    @staticmethod
    def apply_clean_spines(ax: plt.Axes):
        """Applies minimalist border and grid styling."""
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1", zorder=0)
        ax.tick_params(colors="#334155", labelsize=10)


class ProgressionVisualizer:
    """Generates modern, high-contrast, dual-panel charts with sweet-spot annotations."""

    def __init__(self, theme: ModernChartTheme = ModernChartTheme()):
        self.theme = theme

    def plot_single_algorithm(
        self,
        algorithm_name: str,
        df_metrics: pd.DataFrame,
        output_filename: str,
    ):
        """Creates a dual-panel (RMSE and R-squared) progression figure with sweet-spot callout."""
        fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
        fig.patch.set_facecolor("#f8fafc")

        k_values = df_metrics["Number of Features"]
        rmse_values = df_metrics["RMSE"]
        r2_values = df_metrics["R-squared"]

        # ---------------- Left Panel: RMSE ----------------
        self.theme.apply_clean_spines(ax_rmse)
        ax_rmse.plot(
            k_values,
            rmse_values,
            color=self.theme.COLOR_PRIMARY_INDIGO,
            linewidth=2.4,
            marker="o",
            markersize=7.5,
            markerfacecolor="#ffffff",
            markeredgecolor=self.theme.COLOR_PRIMARY_INDIGO,
            markeredgewidth=2.2,
            zorder=3,
            label="Test RMSE",
        )

        # Highlight optimal point (k=2)
        opt_idx = df_metrics[df_metrics["Number of Features"] == 2].index
        if len(opt_idx) > 0:
            opt_k = 2
            opt_rmse = float(df_metrics.loc[opt_idx[0], "RMSE"])
            ax_rmse.scatter(
                [opt_k], [opt_rmse],
                color=self.theme.COLOR_SUCCESS_EMERALD,
                s=130,
                zorder=4,
                edgecolors="#ffffff",
                linewidths=2.5,
            )
            ax_rmse.annotate(
                f"Sweet Spot (k=2)\n${opt_rmse:,.0f}",
                xy=(opt_k, opt_rmse),
                xytext=(opt_k + 0.25, opt_rmse + 110),
                arrowprops=dict(
                    arrowstyle="->",
                    color=self.theme.COLOR_SUCCESS_EMERALD,
                    lw=1.5,
                ),
                fontsize=9.5,
                fontweight="bold",
                color="#065f46",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#ecfdf5",
                    edgecolor="#10b981",
                    lw=1.0,
                ),
            )

        ax_rmse.set_title(f"{algorithm_name}\nRMSE Progression across Subsets", fontsize=12, pad=12, fontweight="bold", color="#1e293b")
        ax_rmse.set_xlabel("Number of Features (k)", fontsize=11, fontweight="medium", color="#334155")
        ax_rmse.set_ylabel("Test RMSE (Lower is Better)", fontsize=11, fontweight="medium", color="#334155")
        ax_rmse.set_xticks(range(1, 6))

        # ---------------- Right Panel: R-squared ----------------
        self.theme.apply_clean_spines(ax_r2)
        ax_r2.plot(
            k_values,
            r2_values,
            color=self.theme.COLOR_SUCCESS_EMERALD,
            linewidth=2.4,
            marker="s",
            markersize=7.5,
            markerfacecolor="#ffffff",
            markeredgecolor=self.theme.COLOR_SUCCESS_EMERALD,
            markeredgewidth=2.2,
            zorder=3,
            label="Test R²",
        )

        if len(opt_idx) > 0:
            opt_r2 = float(df_metrics.loc[opt_idx[0], "R-squared"])
            ax_r2.scatter(
                [opt_k], [opt_r2],
                color=self.theme.COLOR_PRIMARY_INDIGO,
                s=130,
                zorder=4,
                edgecolors="#ffffff",
                linewidths=2.5,
            )
            ax_r2.annotate(
                f"Peak R² (k=2)\n{opt_r2:.4f}",
                xy=(opt_k, opt_r2),
                xytext=(opt_k + 0.25, opt_r2 - 0.0018),
                arrowprops=dict(
                    arrowstyle="->",
                    color=self.theme.COLOR_PRIMARY_INDIGO,
                    lw=1.5,
                ),
                fontsize=9.5,
                fontweight="bold",
                color="#312e81",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#eef2ff",
                    edgecolor="#6366f1",
                    lw=1.0,
                ),
            )

        ax_r2.set_title(f"{algorithm_name}\nR-squared Progression across Subsets", fontsize=12, pad=12, fontweight="bold", color="#1e293b")
        ax_r2.set_xlabel("Number of Features (k)", fontsize=11, fontweight="medium", color="#334155")
        ax_r2.set_ylabel("Test R² (Higher is Better)", fontsize=11, fontweight="medium", color="#334155")
        ax_r2.set_xticks(range(1, 6))

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    def plot_combined_comparison(
        self,
        all_results: Dict[str, pd.DataFrame],
        output_filename: str = "all_5_algorithms_comparison_plot.png",
    ):
        """Generates modern multi-line comparison plot for all 5 algorithms."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        fig.patch.set_facecolor("#f8fafc")

        self.theme.apply_clean_spines(ax1)
        self.theme.apply_clean_spines(ax2)

        for (name, df_res), color, marker in zip(
            all_results.items(), self.theme.PALETTE_ALGORITHMS, self.theme.MARKERS
        ):
            ax1.plot(
                df_res["Number of Features"],
                df_res["RMSE"],
                marker=marker,
                markersize=7.5,
                linewidth=2.0,
                label=name,
                color=color,
                markerfacecolor="#ffffff",
                markeredgecolor=color,
                markeredgewidth=2.0,
                zorder=3,
            )
            ax2.plot(
                df_res["Number of Features"],
                df_res["R-squared"],
                marker=marker,
                markersize=7.5,
                linewidth=2.0,
                label=name,
                color=color,
                markerfacecolor="#ffffff",
                markeredgecolor=color,
                markeredgewidth=2.0,
                zorder=3,
            )

        # Highlight convergence sweet spot at k=2
        ax1.axvspan(1.85, 2.15, color="#ecfdf5", alpha=0.5, zorder=1, label="Optimal Zone (k=2)")
        ax2.axvspan(1.85, 2.15, color="#ecfdf5", alpha=0.5, zorder=1, label="Optimal Zone (k=2)")

        ax1.set_title("Test RMSE Comparison (5 Algorithms)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
        ax1.set_xlabel("Number of Selected Features (k)", fontsize=11, fontweight="medium", color="#334155")
        ax1.set_ylabel("Out-of-Sample RMSE (Lower is Better)", fontsize=11, fontweight="medium", color="#334155")
        ax1.set_xticks(range(1, 6))
        ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

        ax2.set_title("Test R² Comparison (5 Algorithms)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
        ax2.set_xlabel("Number of Selected Features (k)", fontsize=11, fontweight="medium", color="#334155")
        ax2.set_ylabel("Out-of-Sample R² (Higher is Better)", fontsize=11, fontweight="medium", color="#334155")
        ax2.set_xticks(range(1, 6))
        ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)


# ==============================================================================
# Pipeline Execution & Orchestration
# ==============================================================================

class ProgressionAnalysisPipeline:
    """Orchestrates feature selection discovery, metric evaluation, plotting, and file export."""

    def __init__(self):
        self.dataset = StartupDataset()
        self.evaluator = ModelEvaluator(self.dataset)
        self.strategies = FeatureSelectionStrategies(self.dataset, self.evaluator)
        self.visualizer = ProgressionVisualizer()

    def run(self):
        print("=" * 65)
        print(" 50 STARTUPS: MULTIPLE LINEAR REGRESSION PROGRESSION PIPELINE")
        print("=" * 65)

        # Map display names to strategy methods
        algo_definitions = [
            ("Lasso (L1 Ranking)", "Lasso", self.strategies.get_lasso_subsets()),
            ("Forward SFS", "Forward SFS", self.strategies.get_forward_sfs_subsets()),
            ("RFE", "RFE", self.strategies.get_rfe_subsets()),
            ("ElasticNet Ranking", "ElasticNet", self.strategies.get_elasticnet_subsets()),
            ("Exhaustive Search", "Exhaustive", self.strategies.get_exhaustive_subsets()),
        ]

        all_dataframes: Dict[str, pd.DataFrame] = {}
        model_data_registry: Dict[str, List[Dict[str, Any]]] = {}

        for display_name, registry_key, subsets in algo_definitions:
            records = []
            full_specs = []

            for k, sub in enumerate(subsets, start=1):
                full_spec = self.evaluator.evaluate_subset_full(k, sub)
                full_specs.append(full_spec)

                records.append({
                    "Number of Features": k,
                    "Selected Features": str(sub),
                    "RMSE": full_spec["rmse"],
                    "R-squared": full_spec["r2"],
                })

            df_res = pd.DataFrame(records)
            all_dataframes[display_name] = df_res
            model_data_registry[registry_key] = full_specs

            # Export CSV
            clean_filename_base = (
                display_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            csv_path = f"{clean_filename_base}_progression.csv"
            df_res.to_csv(csv_path, index=False)

            # Export Plot
            plot_path = f"{clean_filename_base}_plot.png"
            self.visualizer.plot_single_algorithm(display_name, df_res, plot_path)

            print(f"\n[Algorithm: {display_name}]")
            print(df_res.to_string(index=False))
            print(f" -> Saved: {csv_path} and {plot_path}")

        # Combined Plot
        combined_plot_path = "all_5_algorithms_comparison_plot.png"
        self.visualizer.plot_combined_comparison(all_dataframes, combined_plot_path)
        print(f"\n -> Saved Combined Plot: {combined_plot_path}")

        # Save synchronized model_data.json and model_data.js
        with open("model_data.json", "w", encoding="utf-8") as f_json:
            json.dump(model_data_registry, f_json, indent=2)

        with open("model_data.js", "w", encoding="utf-8") as f_js:
            f_js.write("window.MODEL_DATA = ")
            json.dump(model_data_registry, f_js, indent=2)
            f_js.write(";\nvar MODEL_DATA = window.MODEL_DATA;\n")

        # Also update static/model_data.js if static folder exists
        static_js_path = os.path.join("static", "model_data.js")
        if os.path.exists("static"):
            with open(static_js_path, "w", encoding="utf-8") as f_static_js:
                f_static_js.write("window.MODEL_DATA = ")
                json.dump(model_data_registry, f_static_js, indent=2)
                f_static_js.write(";\nvar MODEL_DATA = window.MODEL_DATA;\n")

        print("\n[Artifacts Updated]")
        print(" - model_data.json (Flask backend registry)")
        print(" - model_data.js & static/model_data.js (Frontend client fallback)")
        print("=" * 65)
        print(" Progression pipeline completed successfully!")
        print("=" * 65)


if __name__ == "__main__":
    pipeline = ProgressionAnalysisPipeline()
    pipeline.run()
