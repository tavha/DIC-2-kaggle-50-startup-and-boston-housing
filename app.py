"""
50 Startups: Flask Web Application & REST API
=============================================
Interactive web platform & RESTful microservice providing live startup profitability
inference and model diagnostics across 5 feature selection algorithms.

Endpoints:
  - GET  /              : Renders the StatsUnlock-style regression explorer
  - GET  /api/models    : Returns model specifications, coefficients, and metrics
  - POST /api/predict   : Computes live profit prediction and variable breakdown
  - GET  /api/summary   : Returns summary comparison data across all models
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, render_template, request, jsonify, Response

# ==============================================================================
# Logging & Application Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger("StartupPredictorApp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DATA_FILE = os.path.join(BASE_DIR, "model_data.json")


# ==============================================================================
# Model Repository Service
# ==============================================================================

class ModelRepository:
    """Manages persistence and retrieval of precalculated regression models."""

    def __init__(self, data_path: str = MODEL_DATA_FILE):
        self.data_path = data_path
        self._registry: Dict[str, List[Dict[str, Any]]] = self._load_registry()

    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.data_path):
            logger.warning("Model data registry file not found at: %s", self.data_path)
            return {}
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Loaded model registry with %d algorithms.", len(data))
                return data
        except Exception as exc:
            logger.error("Failed to parse model registry JSON: %s", exc)
            return {}

    @property
    def registry(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._registry

    def get_algorithm_models(self, algo_name: str) -> Optional[List[Dict[str, Any]]]:
        return self._registry.get(algo_name)

    def get_model_spec(self, algo_name: str, k: int) -> Optional[Dict[str, Any]]:
        models = self.get_algorithm_models(algo_name)
        if not models or k < 1 or k > len(models):
            return None
        return models[k - 1]


# ==============================================================================
# Live Profit Inference Service
# ==============================================================================

class ProfitInferenceService:
    """Calculates predictions and feature-level contribution breakdowns."""

    VALID_STATES = {"California", "Florida", "New York"}

    def __init__(self, repo: ModelRepository):
        self.repo = repo

    def calculate_prediction(
        self,
        algo: str,
        k: int,
        rd_spend: float,
        admin_spend: float,
        marketing_spend: float,
        state: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        model_spec = self.repo.get_model_spec(algo, k)
        if not model_spec:
            return None, f"Unsupported algorithm '{algo}' or invalid subset size k={k}"

        state_clean = state.strip().title() if state else "California"
        if state_clean not in self.VALID_STATES:
            state_clean = "California"

        # Feature value resolution dictionary
        feature_values = {
            "R&D Spend": rd_spend,
            "Administration": admin_spend,
            "Marketing Spend": marketing_spend,
            "State_Florida": 1.0 if state_clean == "Florida" else 0.0,
            "State_New York": 1.0 if state_clean == "New York" else 0.0,
        }

        total_predicted = 0.0
        breakdown: List[Dict[str, Any]] = []

        for item in model_spec.get("coefficients", []):
            feat_name = item["feature"]
            coef_val = float(item["coef"])

            if "Intercept" in feat_name:
                total_predicted += coef_val
                breakdown.append({
                    "feature": "Intercept",
                    "value": 1.0,
                    "coef": coef_val,
                    "contribution": coef_val,
                })
            elif feat_name in feature_values:
                val = feature_values[feat_name]
                contrib = coef_val * val
                total_predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib,
                })

        response_payload = {
            "status": "success",
            "algorithm": algo,
            "k": k,
            "active_features": model_spec.get("features", []),
            "predicted_profit": total_predicted,
            "rmse": model_spec.get("rmse", 0.0),
            "r2": model_spec.get("r2", 0.0),
            "breakdown": breakdown,
        }
        return response_payload, None


# ==============================================================================
# Flask Application Setup
# ==============================================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

model_repo = ModelRepository()
inference_service = ProfitInferenceService(model_repo)


# ==============================================================================
# Web & API Routes
# ==============================================================================

@app.route("/", methods=["GET"])
def index() -> str:
    """Serves the interactive StatsUnlock-style dashboard."""
    return render_template("index.html", initial_algo="Lasso", initial_k=2)


@app.route("/api/models", methods=["GET"])
def get_models() -> Response:
    """Returns the full 5-algorithm feature selection model registry."""
    requested_algo = request.args.get("algo")
    if requested_algo:
        models = model_repo.get_algorithm_models(requested_algo)
        if models is not None:
            return jsonify({requested_algo: models})
        return jsonify({"error": f"Algorithm '{requested_algo}' not found"}), 404
    return jsonify(model_repo.registry)


@app.route("/api/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    """
    Computes forecasted profitability and contribution decomposition.
    Accepts JSON body:
      {
        "algo": "Lasso",
        "k": 2,
        "rd_spend": 100000,
        "admin_spend": 120000,
        "marketing_spend": 150000,
        "state": "California"
      }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        algo = str(data.get("algo", "Lasso"))
        k = int(data.get("k", 2))
        rd_spend = float(data.get("rd_spend", 100000.0))
        admin_spend = float(data.get("admin_spend", 120000.0))
        marketing_spend = float(data.get("marketing_spend", 150000.0))
        state = str(data.get("state", "California"))

        result, err_msg = inference_service.calculate_prediction(
            algo=algo,
            k=k,
            rd_spend=rd_spend,
            admin_spend=admin_spend,
            marketing_spend=marketing_spend,
            state=state,
        )

        if err_msg or not result:
            return jsonify({"status": "error", "message": err_msg}), 400

        return jsonify(result), 200

    except ValueError as val_err:
        return jsonify({"status": "error", "message": f"Invalid numerical format: {val_err}"}), 400
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/summary", methods=["GET"])
def summary() -> Response:
    """Returns comparative metrics across all 5 algorithms and subset sizes."""
    summary_records = []
    for algo_name, models in model_repo.registry.items():
        for spec in models:
            summary_records.append({
                "algorithm": algo_name,
                "k": spec.get("k"),
                "features": spec.get("features", []),
                "rmse": spec.get("rmse"),
                "r2": spec.get("r2"),
                "mae": spec.get("mae"),
            })
    return jsonify(summary_records)


# ==============================================================================
# Development Server Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting 50 Startups Flask Web Service on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=True)
