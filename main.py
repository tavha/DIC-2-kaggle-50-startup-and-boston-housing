"""
50 Startups: CRISP-DM Multiple Linear Regression Pipeline
==========================================================
End-to-end CRISP-DM machine learning pipeline for startup profitability forecasting.
Compares Baseline OLS against 5 feature selection strategies:
  1. LassoCV (L1 Regularization Feature Selection)
  2. RFECV (Recursive Feature Elimination with Cross-Validation)
  3. Sequential Forward Selection (Forward SFS)
  4. ElasticNetCV (L1 + L2 Regularization Feature Selection)
  5. Exhaustive Feature Subset Search

Performs:
  - 5-Fold Cross-Validation evaluation on training set
  - Out-of-sample held-out test evaluation (RMSE, MAE, R², Adjusted R²)
  - Multicollinearity diagnostics via Variance Inflation Factor (VIF)
  - Top 10 largest absolute residual outlier identification
  - Deployment-phase inference simulation
  - Modern publication-grade visualization generation:
      * feature_selection_comparison.png
      * actual_vs_predicted.png
      * top10_outliers_plot.png
"""

import os
import sys
import itertools
from typing import Dict, List, Tuple, Any, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# ==============================================================================
# Global Configuration & Seeds
# ==============================================================================

RANDOM_STATE = 42
CV_FOLDS = 5
KFOLD = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

LOCAL_DATA_FILE = "50_Startups.csv"
REMOTE_DATA_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)

PREDICTOR_VARS = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York",
]
TARGET_VAR = "Profit"


# ==============================================================================
# Phase 2 & 3: Data Understanding & Preparation Module
# ==============================================================================

class StartupDataWarehouse:
    """Handles dataset loading, one-hot dummy transformation, train/test splitting, and scaling."""

    def __init__(self, filepath: str = LOCAL_DATA_FILE, url: str = REMOTE_DATA_URL):
        self.raw_df = self._fetch_dataset(filepath, url)
        self.feature_names = list(PREDICTOR_VARS)
        self.target_name = TARGET_VAR

        # California is reference baseline (dropped)
        self.encoded_df = pd.get_dummies(self.raw_df, columns=["State"], dtype=int)
        self.X = self.encoded_df[self.feature_names].astype(float)
        self.y = self.encoded_df[self.target_name].astype(float)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=RANDOM_STATE
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index,
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index,
        )

    @staticmethod
    def _fetch_dataset(filepath: str, url: str) -> pd.DataFrame:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        try:
            return pd.read_csv(url)
        except Exception as err:
            raise RuntimeError(f"Failed to retrieve 50 Startups dataset: {err}") from err


# ==============================================================================
# Model Evaluation & Diagnostics Module
# ==============================================================================

class ModelDiagnostics:
    """Calculates cross-validation, out-of-sample metrics, VIF, and residuals."""

    @staticmethod
    def compute_regression_metrics(
        y_actual: pd.Series, y_predicted: np.ndarray, num_features: int
    ) -> Dict[str, float]:
        n = len(y_actual)
        r2 = float(r2_score(y_actual, y_predicted))
        mae = float(mean_absolute_error(y_actual, y_predicted))
        mse = float(mean_squared_error(y_actual, y_predicted))
        rmse = float(np.sqrt(mse))

        adj_r2 = (
            float(1.0 - (1.0 - r2) * (n - 1) / (n - num_features - 1))
            if n > num_features + 1
            else float("nan")
        )

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adj_r2,
        }

    @staticmethod
    def compute_vif_dataframe(X_matrix: pd.DataFrame, feature_subset: List[str]) -> pd.DataFrame:
        """Calculates Variance Inflation Factor for the selected feature set."""
        X_const = sm.add_constant(X_matrix[feature_subset], has_constant="add")
        vif_records = []
        for idx, col in enumerate(X_const.columns):
            val = variance_inflation_factor(X_const.values, idx)
            vif_records.append({"Feature": col, "VIF": round(val, 3)})
        return pd.DataFrame(vif_records)

    @staticmethod
    def extract_top_outliers(
        raw_df: pd.DataFrame,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        fitted_model: LinearRegression,
        feature_subset: List[str],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Descriptive full-sample residual analysis identifying largest deviations."""
        df_out = raw_df.copy()
        df_out["Predicted_Profit"] = fitted_model.predict(X_full[feature_subset])
        df_out["Residual"] = df_out["Profit"] - df_out["Predicted_Profit"]
        df_out["Absolute_Residual"] = df_out["Residual"].abs()

        top_df = df_out.nlargest(top_n, "Absolute_Residual").copy()
        top_df.insert(0, "Rank", np.arange(1, len(top_df) + 1))
        top_df.insert(1, "Original_Index", top_df.index)
        return top_df


# ==============================================================================
# Phase 4: Feature Selection Search Engine
# ==============================================================================

class FeatureSelectionSuite:
    """Encapsulates 5 distinct feature selection algorithms plus baseline OLS."""

    def __init__(self, data: StartupDataWarehouse):
        self.data = data

    def run_baseline_ols(self) -> List[str]:
        return list(self.data.feature_names)

    def run_lasso_cv(self) -> List[str]:
        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=KFOLD, max_iter=100000, random_state=RANDOM_STATE)),
        ])
        lasso_pipe.fit(self.data.X_train, self.data.y_train)
        coefficients = lasso_pipe.named_steps["lasso"].coef_

        selected = [
            feat for feat, coef in zip(self.data.feature_names, coefficients)
            if abs(coef) > 1e-8
        ]
        return selected or [self.data.feature_names[np.argmax(np.abs(coefficients))]]

    def run_rfecv(self) -> List[str]:
        rfecv = RFECV(
            estimator=LinearRegression(),
            step=1,
            cv=KFOLD,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=1,
        )
        rfecv_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", rfecv),
        ])
        rfecv_pipe.fit(self.data.X_train, self.data.y_train)
        return list(self.data.X_train.columns[rfecv_pipe.named_steps["selector"].support_])

    def run_forward_sfs(self) -> List[str]:
        best_rmse = float("inf")
        best_features = None

        for k in range(1, len(self.data.feature_names) + 1):
            if k == len(self.data.feature_names):
                candidate = list(self.data.feature_names)
            else:
                sfs = SequentialFeatureSelector(
                    estimator=LinearRegression(),
                    n_features_to_select=k,
                    direction="forward",
                    scoring="neg_root_mean_squared_error",
                    cv=KFOLD,
                )
                sfs.fit(self.data.X_train, self.data.y_train)
                candidate = list(self.data.X_train.columns[sfs.get_support()])

            cv_score = -cross_val_score(
                LinearRegression(),
                self.data.X_train[candidate],
                self.data.y_train,
                cv=KFOLD,
                scoring="neg_root_mean_squared_error",
            ).mean()

            if cv_score < best_rmse:
                best_rmse = cv_score
                best_features = candidate

        return best_features or list(self.data.feature_names)

    def run_elasticnet_cv(self) -> List[str]:
        elastic_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("elastic", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                cv=KFOLD,
                max_iter=100000,
                random_state=RANDOM_STATE,
            )),
        ])
        elastic_pipe.fit(self.data.X_train, self.data.y_train)
        coefficients = elastic_pipe.named_steps["elastic"].coef_

        selected = [
            feat for feat, coef in zip(self.data.feature_names, coefficients)
            if abs(coef) > 1e-8
        ]
        return selected or [self.data.feature_names[np.argmax(np.abs(coefficients))]]

    def run_exhaustive_search(self) -> List[str]:
        best_score = float("inf")
        best_subset = None

        for k in range(1, len(self.data.feature_names) + 1):
            for subset in itertools.combinations(self.data.feature_names, k):
                sub_list = list(subset)
                score = -cross_val_score(
                    LinearRegression(),
                    self.data.X_train[sub_list],
                    self.data.y_train,
                    cv=KFOLD,
                    scoring="neg_root_mean_squared_error",
                ).mean()

                if score < best_score:
                    best_score = score
                    best_subset = sub_list

        return best_subset or list(self.data.feature_names)


# ==============================================================================
# Phase 5: Modern Visualization Engine
# ==============================================================================

class CrispDMVisualizer:
    """Generates modern, publication-grade figures for CRISP-DM reporting."""

    @staticmethod
    def apply_canvas_styling(ax: plt.Axes):
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1", zorder=0)
        ax.tick_params(colors="#334155", labelsize=10)

    @classmethod
    def plot_algorithm_comparison(
        cls,
        results_df: pd.DataFrame,
        output_filename: str = "feature_selection_comparison.png",
    ):
        """Generates modern horizontal bar chart comparing CV RMSE across methods."""
        fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
        fig.patch.set_facecolor("#f8fafc")
        cls.apply_canvas_styling(ax)

        df_sorted = results_df.sort_values("CV_RMSE", ascending=True).reset_index(drop=True)
        y_positions = np.arange(len(df_sorted))

        # Color scheme: Best model is emerald, baseline is slate, others are indigo
        best_rmse = df_sorted.iloc[0]["CV_RMSE"]
        bar_colors = []
        for _, row in df_sorted.iterrows():
            if row["CV_RMSE"] == best_rmse:
                bar_colors.append("#059669")  # Emerald for top performer
            elif "Baseline" in row["Method"]:
                bar_colors.append("#64748b")  # Slate for baseline
            else:
                bar_colors.append("#4f46e5")  # Indigo for other algorithms

        bars = ax.barh(
            y_positions,
            df_sorted["CV_RMSE"],
            color=bar_colors,
            height=0.6,
            edgecolor="#ffffff",
            linewidth=1.2,
            zorder=3,
        )

        # Annotate values on bars
        for bar, val in zip(bars, df_sorted["CV_RMSE"]):
            ax.text(
                val + 120,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.2f}",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="#1e293b",
            )

        # Baseline reference line
        baseline_val = float(df_sorted[df_sorted["Method"].str.contains("Baseline")]["CV_RMSE"].iloc[0])
        ax.axvline(
            baseline_val,
            color="#ef4444",
            linestyle=":",
            linewidth=1.8,
            zorder=2,
            label=f"Baseline OLS (${baseline_val:,.2f})",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_sorted["Method"], fontsize=10.5, fontweight="medium", color="#1e293b")
        ax.invert_yaxis()
        ax.set_xlabel("Cross-Validated RMSE in USD (Lower is Better)", fontsize=11, fontweight="medium", color="#334155")
        ax.set_title("50 Startups: Feature Selection Method Comparison\n5-Fold Cross-Validation Error", fontsize=13, fontweight="bold", pad=14, color="#0f172a")
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="lower right", fontsize=9.5)

        max_val = df_sorted["CV_RMSE"].max()
        ax.set_xlim(0, max_val * 1.15)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    @classmethod
    def plot_actual_vs_predicted(
        cls,
        y_test: pd.Series,
        test_pred: np.ndarray,
        best_method_name: str,
        output_filename: str = "actual_vs_predicted.png",
    ):
        """Generates modern actual vs predicted parity scatter plot on test set."""
        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
        fig.patch.set_facecolor("#f8fafc")
        cls.apply_canvas_styling(ax)

        y_actual_k = y_test / 1000.0
        y_pred_k = test_pred / 1000.0

        min_val = min(y_actual_k.min(), y_pred_k.min()) * 0.95
        max_val = max(y_actual_k.max(), y_pred_k.max()) * 1.05

        # Parity line
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="#ef4444",
            linewidth=2.0,
            label="Ideal Prediction Line (y = x)",
            zorder=2,
        )

        # Scatter points
        ax.scatter(
            y_actual_k,
            y_pred_k,
            color="#4f46e5",
            s=100,
            edgecolors="#ffffff",
            linewidths=2.0,
            alpha=0.9,
            zorder=4,
            label="Held-Out Test Startups (N=10)",
        )

        # Performance badge
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        badge_text = f"Model: {best_method_name}\nTest RMSE: ${test_rmse:,.2f}\nTest R²: {test_r2:.4f}"
        ax.text(
            0.05,
            0.92,
            badge_text,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            color="#1e293b",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                lw=1.2,
            ),
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("Actual Startup Profit ($ thousands)", fontsize=11, fontweight="medium", color="#334155")
        ax.set_ylabel("Predicted Startup Profit ($ thousands)", fontsize=11, fontweight="medium", color="#334155")
        ax.set_title(f"Actual vs. Predicted Profit: {best_method_name}\nEvaluated on Out-of-Sample Test Set", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="lower right", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    @classmethod
    def plot_top10_outliers(
        cls,
        raw_df: pd.DataFrame,
        top10_df: pd.DataFrame,
        best_method_name: str,
        output_filename: str = "top10_outliers_plot.png",
    ):
        """Generates modern state-clustered scatter plot with top 10 residual callouts."""
        fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=300)
        fig.patch.set_facecolor("#f8fafc")
        cls.apply_canvas_styling(ax)

        state_color_map = {
            "California": "#4f46e5",   # Indigo
            "Florida": "#059669",      # Emerald
            "New York": "#d97706",     # Amber
        }

        # Plot all 50 samples by state
        for state, color in state_color_map.items():
            state_subset = raw_df[raw_df["State"] == state]
            ax.scatter(
                state_subset["Profit"] / 1000.0,
                state_subset["Predicted_Profit"] / 1000.0,
                label=f"State: {state}",
                color=color,
                s=80,
                alpha=0.75,
                edgecolors="#ffffff",
                linewidths=1.5,
                zorder=3,
            )

        # Range bounds
        min_v = min(raw_df["Profit"].min(), raw_df["Predicted_Profit"].min()) / 1000.0 * 0.95
        max_v = max(raw_df["Profit"].max(), raw_df["Predicted_Profit"].max()) / 1000.0 * 1.05

        # Parity line
        ax.plot(
            [min_v, max_v],
            [min_v, max_v],
            linestyle="--",
            color="#94a3b8",
            linewidth=1.8,
            label="Perfect Parity (y = x)",
            zorder=2,
        )

        # Scatter regression trendline
        fit_coef = np.polyfit(
            raw_df["Profit"] / 1000.0, raw_df["Predicted_Profit"] / 1000.0, 1
        )
        xs = np.linspace(min_v, max_v, 200)
        ax.plot(
            xs,
            fit_coef[0] * xs + fit_coef[1],
            color="#6366f1",
            linewidth=2.0,
            label="Linear Trendline",
            zorder=2,
        )

        # Highlight Top 10 absolute residuals
        for _, row in top10_df.iterrows():
            px = row["Profit"] / 1000.0
            py = row["Predicted_Profit"] / 1000.0
            rank = int(row["Rank"])

            # Outer ring marker
            ax.scatter(
                px, py,
                s=260,
                facecolors="none",
                edgecolors="#e11d48",
                linewidths=2.2,
                zorder=5,
            )

            # Styled numbered pill label
            ax.annotate(
                f"#{rank}",
                xy=(px, py),
                xytext=(8, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#9f1239",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="#ffe4e6",
                    edgecolor="#f43f5e",
                    lw=1.0,
                ),
                zorder=6,
            )

        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_xlabel("Actual Profit ($ thousands)", fontsize=11, fontweight="medium", color="#334155")
        ax.set_ylabel("Predicted Profit ($ thousands)", fontsize=11, fontweight="medium", color="#334155")
        ax.set_title(
            f"50 Startups: Residual Diagnostics ({best_method_name})\n"
            "Top 10 Largest Absolute Residuals Highlighted",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="lower right", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)


# ==============================================================================
# Phase 6: Production Inference Engine
# ==============================================================================

class StartupInferenceEngine:
    """Production deployment predictor for incoming venture profiles."""

    def __init__(self, model: LinearRegression, active_features: List[str]):
        self.model = model
        self.active_features = active_features

    def predict_venture(self, venture_spec: Dict[str, Any]) -> float:
        row_df = pd.DataFrame([venture_spec])
        pred = self.model.predict(row_df[self.active_features])[0]
        return float(pred)


# ==============================================================================
# Master CRISP-DM Workflow Controller
# ==============================================================================

class CrispDMPipeline:
    """Master controller orchestrating the CRISP-DM lifecycle for 50 Startups."""

    def __init__(self):
        self.data_wh: Optional[StartupDataWarehouse] = None
        self.selection_suite: Optional[FeatureSelectionSuite] = None
        self.results_registry: List[Dict[str, Any]] = []
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.best_method: str = ""
        self.best_features: List[str] = []
        self.best_model: Optional[LinearRegression] = None

    def execute(self):
        self._phase1_business_understanding()
        self._phase2_data_understanding()
        self._phase3_data_preparation()
        self._phase4_modeling()
        self._phase5_evaluation()
        self._phase6_deployment()

    def _phase1_business_understanding(self):
        print("\n" + "=" * 65)
        print("PHASE 1: BUSINESS UNDERSTANDING")
        print("=" * 65)
        print("Objective: Predict startup profitability from financial allocations")
        print("Methodology: Benchmark OLS Multiple Linear Regression against")
        print("             5 distinct feature selection algorithms under CRISP-DM.")

    def _phase2_data_understanding(self):
        print("\n" + "=" * 65)
        print("PHASE 2: DATA UNDERSTANDING")
        print("=" * 65)
        self.data_wh = StartupDataWarehouse()
        print(f"Dataset Shape: {self.data_wh.raw_df.shape[0]} rows x {self.data_wh.raw_df.shape[1]} columns")
        print("\nSample Data (First 5 records):")
        print(self.data_wh.raw_df.head())
        print("\nMissing Values Count:")
        print(self.data_wh.raw_df.isna().sum())

    def _phase3_data_preparation(self):
        print("\n" + "=" * 65)
        print("PHASE 3: DATA PREPARATION")
        print("=" * 65)
        print("Encoding: One-hot encoded 'State'; 'California' retained as reference baseline.")
        print("Features:", self.data_wh.feature_names)
        print(f"Training Observations (80%): {len(self.data_wh.X_train)}")
        print(f"Testing Observations  (20%): {len(self.data_wh.X_test)}")

    def _phase4_modeling(self):
        print("\n" + "=" * 65)
        print("PHASE 4: MODELING (Feature Selection)")
        print("=" * 65)
        self.selection_suite = FeatureSelectionSuite(self.data_wh)

        algorithms = [
            ("Baseline OLS", self.selection_suite.run_baseline_ols),
            ("LassoCV", self.selection_suite.run_lasso_cv),
            ("RFECV", self.selection_suite.run_rfecv),
            ("Forward SFS", self.selection_suite.run_forward_sfs),
            ("ElasticNetCV", self.selection_suite.run_elasticnet_cv),
            ("Exhaustive", self.selection_suite.run_exhaustive_search),
        ]

        for name, algo_func in algorithms:
            selected = algo_func()
            self._evaluate_and_register(name, selected)

    def _evaluate_and_register(self, name: str, selected_features: List[str]):
        sub = list(selected_features)
        model = LinearRegression().fit(self.data_wh.X_train[sub], self.data_wh.y_train)
        preds = model.predict(self.data_wh.X_test[sub])

        cv_rmse = -cross_val_score(
            LinearRegression(),
            self.data_wh.X_train[sub],
            self.data_wh.y_train,
            cv=KFOLD,
            scoring="neg_root_mean_squared_error",
        ).mean()

        metric_dict = ModelDiagnostics.compute_regression_metrics(
            self.data_wh.y_test, preds, len(sub)
        )
        metric_dict.update({
            "Method": name,
            "N_Features": len(sub),
            "Selected_Features": ", ".join(sub),
            "CV_RMSE": cv_rmse,
        })

        self.results_registry.append(metric_dict)
        self.model_registry[name] = {
            "model": model,
            "features": sub,
            "predictions": preds,
        }

        print(f"[{name}]")
        print(f"  Selected Features ({len(sub)}): {sub}")
        print(f"  5-Fold CV RMSE: ${cv_rmse:,.2f}")
        print(f"  Test R^2: {metric_dict['R2']:.4f}")

    def _phase5_evaluation(self):
        print("\n" + "=" * 65)
        print("PHASE 5: EVALUATION")
        print("=" * 65)

        results_df = pd.DataFrame(self.results_registry)[[
            "Method", "N_Features", "Selected_Features",
            "CV_RMSE", "MAE", "MSE", "RMSE", "R2", "Adjusted_R2"
        ]].sort_values("CV_RMSE").reset_index(drop=True)

        print("\n=== MODEL COMPARISON TABLE ===")
        print(results_df.round(4).to_string(index=False))

        # Save CSV comparison
        comparison_csv = "feature_selection_comparison.csv"
        results_df.to_csv(comparison_csv, index=False)
        print(f"\nSaved model comparison to: {comparison_csv}")

        # Best performer
        self.best_method = str(results_df.iloc[0]["Method"])
        self.best_features = self.model_registry[self.best_method]["features"]
        self.best_model = self.model_registry[self.best_method]["model"]

        print(f"\nBest Performing Method (by CV RMSE): {self.best_method}")
        print(f"Optimal Feature Subset: {self.best_features}")

        # VIF multicollinearity check
        print("\n=== VARIANCE INFLATION FACTOR (VIF) DIAGNOSTICS ===")
        vif_df = ModelDiagnostics.compute_vif_dataframe(self.data_wh.X_train, self.best_features)
        print(vif_df.to_string(index=False))

        # Plot 1: Feature selection comparison
        comparison_plot = "feature_selection_comparison.png"
        CrispDMVisualizer.plot_algorithm_comparison(results_df, comparison_plot)
        print(f"Saved: {comparison_plot}")

        # Plot 2: Actual vs Predicted
        test_pred = self.best_model.predict(self.data_wh.X_test[self.best_features])
        actual_vs_pred_plot = "actual_vs_predicted.png"
        CrispDMVisualizer.plot_actual_vs_predicted(
            self.data_wh.y_test, test_pred, self.best_method, actual_vs_pred_plot
        )
        print(f"Saved: {actual_vs_pred_plot}")

        # Refit best model on full data for descriptive residual analysis
        full_model = LinearRegression().fit(self.data_wh.X[self.best_features], self.data_wh.y)
        self.data_wh.raw_df["Predicted_Profit"] = full_model.predict(self.data_wh.X[self.best_features])
        top10_df = ModelDiagnostics.extract_top_outliers(
            self.data_wh.raw_df, self.data_wh.X, self.data_wh.y, full_model, self.best_features
        )

        print("\n=== TOP 10 LARGEST ABSOLUTE RESIDUALS ===")
        print(top10_df[[
            "Rank", "Original_Index", "State", "Profit",
            "Predicted_Profit", "Residual", "Absolute_Residual"
        ]].round(2).to_string(index=False))

        outliers_csv = "top10_outliers.csv"
        top10_df.to_csv(outliers_csv, index=False)
        print(f"Saved top 10 outliers table to: {outliers_csv}")

        # Plot 3: Top 10 Outliers
        outliers_plot = "top10_outliers_plot.png"
        CrispDMVisualizer.plot_top10_outliers(
            self.data_wh.raw_df, top10_df, self.best_method, outliers_plot
        )
        print(f"Saved: {outliers_plot}")

        self.full_refit_model = full_model

    def _phase6_deployment(self):
        print("\n" + "=" * 65)
        print("PHASE 6: DEPLOYMENT")
        print("=" * 65)

        engine = StartupInferenceEngine(self.full_refit_model, self.best_features)

        # Simulation: New startup venture in California
        sample_venture = {
            "R&D Spend": 100000.0,
            "Administration": 120000.0,
            "Marketing Spend": 150000.0,
            "State_Florida": 0.0,
            "State_New York": 0.0,
        }

        predicted_val = engine.predict_venture(sample_venture)
        print(f"Simulated New Startup Profile: {sample_venture}")
        print(f"Predicted Annual Profit: ${predicted_val:,.2f}")
        print("\nCRISP-DM Cycle Finished Successfully!")


if __name__ == "__main__":
    pipeline = CrispDMPipeline()
    pipeline.execute()
