#!/usr/bin/env python3
"""Run all experiments in the correct dependency order."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    # Data exploration
    "experiments/01_data_exploration/01_overview_and_filters.py",
    "experiments/01_data_exploration/02_fuel_labels_and_intervals.py",
    "experiments/01_data_exploration/03_traj_quality_and_sources.py",
    # Feature engineering
    "experiments/02_feature_engineering/04_physics_baseline_validation.py",
    "experiments/02_feature_engineering/08_physics_features_v2.py",
    "experiments/02_feature_engineering/09_physics_features_v3.py",
    "experiments/02_feature_engineering/09_mass_features.py",
    # Baselines
    "experiments/03_baselines/05_baseline_modeling.py",
    "experiments/03_baselines/06_physics_ablation.py",
    "experiments/03_baselines/07_sparsity_ablation.py",
    # Hybrid models
    "experiments/04_hybrid_models/07_catboost.py",
    "experiments/04_hybrid_models/10_fuel_flow_target.py",
    "experiments/04_hybrid_models/09_aircraft_experts.py",
    "experiments/04_hybrid_models/10_optuna.py",
    "experiments/04_hybrid_models/13_flow_vs_prc.py",
    # Ensemble
    "experiments/05_ensemble/08_ensemble.py",
    "experiments/05_ensemble/11_stacking.py",
    "experiments/05_ensemble/12_verify_ensemble.py",
    # LOTO / generalization
    "experiments/06_loto_generalization/15_leave_one_type_out.py",
    "experiments/06_loto_generalization/15b_loto_residual_matched.py",
    "experiments/06_loto_generalization/17_loto_significance_and_transfer_distance.py",
    # Gap closing
    "experiments/07_gap_closing/17_official_prc_evaluation.py",
    "experiments/07_gap_closing/18_official_error_analysis.py",
    "experiments/07_gap_closing/19_gap_closing_campaign.py",
    "experiments/07_gap_closing/25_r3_dynamic_mass.py",
    "experiments/07_gap_closing/26_r3_ensemble_mass.py",
    # Interpretability
    "experiments/09_interpretability/14_shap_explainability.py",
]

ENV = {"PYTHONPATH": str(ROOT / "src"), **dict(sys.environ)}

for exp in EXPERIMENTS:
    path = ROOT / exp
    if not path.exists():
        print(f"[SKIP] {exp} — not found")
        continue
    print(f"\n{'='*72}")
    print(f"[RUN] {exp}")
    print(f"{'='*72}")
    result = subprocess.run([sys.executable, str(path)], env=ENV)
    if result.returncode != 0:
        print(f"[FAIL] {exp} — exit code {result.returncode}")
        sys.exit(1)
    print(f"[OK] {exp}")
