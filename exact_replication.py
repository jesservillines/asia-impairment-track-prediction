#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exact Replication Script

This script exactly replicates the process from 03_ensemble_final_training_and_submission.ipynb,
using the same parameters, model setups, and ensemble approach with equal weights.

Author: Jesse Villines
Date: 2025-05-02
"""

# =========================
# 1. Imports & Configuration
# =========================

import numpy as np
import pandas as pd
import joblib

from sklearn.multioutput import MultiOutputRegressor

# CatBoost, XGBoost, HistGradientBoosting
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# 2. Define File Paths
# =========================

# Input files
PROCESSED_TRAIN_PATH = "./data/train_processed.csv"
PROCESSED_TEST_PATH = "./data/test_processed.csv"

BEST_PARAMS_CAT_PATH = "./data/best_params_catboost.pkl"
BEST_PARAMS_XGB_PATH = "./data/best_params_xgb.pkl"
BEST_PARAMS_HGB_PATH = "./data/best_params_hgb.pkl"

# Output files
SUBMISSION_PATH = "./data/submission_exact_replication.csv"

# Model save paths
MODELS_DIR = "./models_exact"
import os
os.makedirs(MODELS_DIR, exist_ok=True)

CATBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "catboost_exact_model.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_exact_model.pkl")
HGB_MODEL_PATH = os.path.join(MODELS_DIR, "hgb_exact_model.pkl")

# =========================
# 3. Main Function
# =========================

def main():
    """Main function to replicate the original notebook exactly"""
    
    print("=" * 60)
    print("REPLICATING ORIGINAL NOTEBOOK PROCESS")
    print("=" * 60)
    
    # =========================
    # 4. Load Processed Train & Test
    # =========================
    
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    
    print("Processed train_df shape:", train_df.shape)
    print("Processed test_df shape :", test_df.shape)
    
    # Store test PIDs
    test_pid = test_df['PID'].values
    
    # Remove PID if it's still in the train_df/test_df as a column
    if 'PID' in train_df.columns:
        train_df.drop(columns=['PID'], inplace=True)
    if 'PID' in test_df.columns:
        test_df.drop(columns=['PID'], inplace=True)
    
    # =========================
    # 5. Identify Targets & Features
    # =========================
    
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Separate X, y for the training set
    X_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    X_train_full = train_df[X_cols].values  # all features
    y_train_full = train_df[TARGET_COLS].values
    
    # For the test set, we only have features
    X_test_full = test_df[X_cols].values
    
    print("X_train_full shape:", X_train_full.shape)
    print("y_train_full shape:", y_train_full.shape)
    print("X_test_full shape :", X_test_full.shape)
    
    # =========================
    # 6. Load Best Hyperparameters
    # =========================
    
    best_params_cat = joblib.load(BEST_PARAMS_CAT_PATH)
    best_params_xgb = joblib.load(BEST_PARAMS_XGB_PATH)
    best_params_hgb = joblib.load(BEST_PARAMS_HGB_PATH)
    
    print("\nBest CatBoost params:\n", best_params_cat)
    print("Best XGBoost params:\n", best_params_xgb)
    print("Best HistGB params:\n", best_params_hgb)
    
    # =========================
    # 7. Build Final Models - EXACTLY as in notebook
    # =========================
    
    # --- 7.1 CatBoost ---
    print("\nTraining CatBoost on full training data...")
    catboost_model = CatBoostRegressor(
        **best_params_cat,
        random_state=42,
        verbose=0
    )
    catboost_multi = MultiOutputRegressor(catboost_model)
    
    # --- 7.2 XGB ---
    print("Training XGBoost on full training data...")
    xgb_model = XGBRegressor(
        **best_params_xgb,
        random_state=42
    )
    xgb_multi = MultiOutputRegressor(xgb_model)
    
    # --- 7.3 HistGradientBoosting ---
    print("Training HistGradientBoosting on full training data...")
    hgb_model = HistGradientBoostingRegressor(
        **best_params_hgb,
        random_state=42
    )
    hgb_multi = MultiOutputRegressor(hgb_model)
    
    # =========================
    # 8. Train Each Model on Entire Training Set
    # =========================
    
    print("Training CatBoost model...")
    catboost_multi.fit(X_train_full, y_train_full)
    
    print("Training XGBoost model...")
    xgb_multi.fit(X_train_full, y_train_full)
    
    print("Training HistGradientBoosting model...")
    hgb_multi.fit(X_train_full, y_train_full)
    
    # =========================
    # 8.1 Save Trained Models
    # =========================
    
    print(f"\nSaving CatBoost model to {CATBOOST_MODEL_PATH}...")
    joblib.dump(catboost_multi, CATBOOST_MODEL_PATH)
    
    print(f"Saving XGBoost model to {XGB_MODEL_PATH}...")
    joblib.dump(xgb_multi, XGB_MODEL_PATH)
    
    print(f"Saving HistGradientBoosting model to {HGB_MODEL_PATH}...")
    joblib.dump(hgb_multi, HGB_MODEL_PATH)
    
    print("All models saved successfully!")
    
    # =========================
    # 9. Generate Predictions & Ensemble - EXACTLY as in notebook
    # =========================
    
    print("\nPredicting on test set...")
    
    pred_cat = catboost_multi.predict(X_test_full)
    pred_xgb = xgb_multi.predict(X_test_full)
    pred_hgb = hgb_multi.predict(X_test_full)
    
    # Ensemble by simple average - EXACTLY as in notebook (equal weights)
    pred_ensemble = (pred_cat + pred_xgb + pred_hgb) / 3.0
    
    # =========================
    # 10. Round & Clip Predictions - EXACTLY as in notebook
    # =========================
    
    # Because targets range from 0..5
    pred_ensemble_rounded = np.round(pred_ensemble)
    pred_ensemble_clipped = np.clip(pred_ensemble_rounded, 0, 5)  # ensures within [0,5]
    pred_ensemble_clipped = pred_ensemble_clipped.astype(int)
    
    # =========================
    # 11. Create Submission DataFrame - EXACTLY as in notebook
    # =========================
    
    # Reconstruct columns in the correct order: [PID, <all target columns>]
    submission_df = pd.DataFrame({
        "PID": test_pid
    })
    
    for i, col in enumerate(TARGET_COLS):
        submission_df[col] = pred_ensemble_clipped[:, i]
    
    # =========================
    # 12. Save Submission
    # =========================
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission file saved to: {SUBMISSION_PATH}")
    
    print("\nAll done! This submission should exactly match the original notebook.")
    print(f"\nTrained models have been saved to the {MODELS_DIR} directory.")
    print("You can use these models for future inference while ensuring exact replication.")

if __name__ == "__main__":
    main()
