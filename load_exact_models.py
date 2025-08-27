#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load Exact Models Script

This script:
1) Loads the models saved by exact_replication.py
2) Loads the processed training and test data
3) Performs validation on the training data
4) Generates predictions on the test data using the exact same ensemble approach
5) Creates and saves a submission file

Author: Jesse Villines
Date: 2025-05-02
"""

# =========================
# 1. Imports & Configuration
# =========================

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =========================
# 2. Define File Paths
# =========================

# Input files
PROCESSED_TRAIN_PATH = "./data/train_processed.csv"
PROCESSED_TEST_PATH = "./data/test_processed.csv"

# Model paths
MODELS_DIR = "./models_exact"
CATBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "catboost_exact_model.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_exact_model.pkl")
HGB_MODEL_PATH = os.path.join(MODELS_DIR, "hgb_exact_model.pkl")

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUBMISSION_PATH = f"./data/submission_loaded_exact_{TIMESTAMP}.csv"

# =========================
# 3. Helper Functions
# =========================

def evaluate_model(model, X, y_true, model_name="Model"):
    """Evaluate a single model's performance"""
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse_per_target = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    mae_per_target = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    
    avg_rmse = np.mean(rmse_per_target)
    avg_mae = np.mean(mae_per_target)
    
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Average MAE : {avg_mae:.4f}")
    
    return y_pred, rmse_per_target, mae_per_target

def evaluate_ensemble(preds, y_true, target_cols):
    """Evaluate ensemble performance with equal weights (exactly as in the original)"""
    # Calculate ensemble prediction (equal weights)
    ensemble_pred = np.mean(preds, axis=0)
    
    # Round and clip
    ensemble_pred_rounded = np.round(ensemble_pred)
    ensemble_pred_clipped = np.clip(ensemble_pred_rounded, 0, 5).astype(int)
    
    # Calculate metrics
    rmse_per_target = np.sqrt(mean_squared_error(y_true, ensemble_pred_clipped, multioutput='raw_values'))
    mae_per_target = mean_absolute_error(y_true, ensemble_pred_clipped, multioutput='raw_values')
    
    avg_rmse = np.mean(rmse_per_target)
    avg_mae = np.mean(mae_per_target)
    
    print("\nEnsemble Performance (equal weights):")
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Average MAE : {avg_mae:.4f}")
    
    # Print per-target metrics
    print("\nPer-target metrics:")
    for i, col in enumerate(target_cols):
        print(f"  {col}: RMSE = {rmse_per_target[i]:.4f}, MAE = {mae_per_target[i]:.4f}")
    
    return ensemble_pred_clipped

def load_models():
    """Load all models from disk"""
    print("Loading models from disk...")
    
    # Check if models exist
    if not all(os.path.exists(path) for path in [CATBOOST_MODEL_PATH, XGB_MODEL_PATH, HGB_MODEL_PATH]):
        raise FileNotFoundError("One or more model files not found. Please run exact_replication.py first.")
    
    # Load models
    catboost_multi = joblib.load(CATBOOST_MODEL_PATH)
    print(f"  Loaded CatBoost model from {CATBOOST_MODEL_PATH}")
    
    xgb_multi = joblib.load(XGB_MODEL_PATH)
    print(f"  Loaded XGBoost model from {XGB_MODEL_PATH}")
    
    hgb_multi = joblib.load(HGB_MODEL_PATH)
    print(f"  Loaded HistGradientBoosting model from {HGB_MODEL_PATH}")
    
    print("All models loaded successfully!")
    
    return catboost_multi, xgb_multi, hgb_multi

def generate_submission(models, X_test, test_pid, target_cols, output_path=SUBMISSION_PATH):
    """Generate predictions using loaded models and save submission file - exactly as in original"""
    catboost_multi, xgb_multi, hgb_multi = models
    
    print("\nGenerating predictions for test set...")
    
    # Generate predictions from each model
    pred_cat = catboost_multi.predict(X_test)
    pred_xgb = xgb_multi.predict(X_test)
    pred_hgb = hgb_multi.predict(X_test)
    
    # Ensemble by simple average - EXACTLY as in the original notebook
    pred_ensemble = (pred_cat + pred_xgb + pred_hgb) / 3.0
    
    # Round and clip predictions
    pred_ensemble_rounded = np.round(pred_ensemble)
    pred_ensemble_clipped = np.clip(pred_ensemble_rounded, 0, 5)
    pred_ensemble_clipped = pred_ensemble_clipped.astype(int)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({"PID": test_pid})
    
    # Add target columns
    for i, col in enumerate(target_cols):
        submission_df[col] = pred_ensemble_clipped[:, i]
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    return submission_df

# =========================
# 4. Main Function
# =========================

def main():
    """Main function to load models, validate, and generate submissions"""
    
    print("=" * 60)
    print("LOADING EXACT MODELS AND PERFORMING VALIDATION")
    print("=" * 60)
    
    # Load data
    print("Loading processed train and test data...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    
    print(f"Processed train_df shape: {train_df.shape}")
    print(f"Processed test_df shape : {test_df.shape}")
    
    # Store test PIDs
    test_pid = test_df['PID'].values
    
    # Remove PID if it's still in the dataframes
    if 'PID' in train_df.columns:
        train_df.drop(columns=['PID'], inplace=True)
    if 'PID' in test_df.columns:
        test_df.drop(columns=['PID'], inplace=True)
    
    # Define target columns
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Separate features and targets
    X_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    X_train_full = train_df[X_cols].values
    y_train_full = train_df[TARGET_COLS].values
    X_test_full = test_df[X_cols].values
    
    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"y_train_full shape: {y_train_full.shape}")
    print(f"X_test_full shape : {X_test_full.shape}")
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"Validation split - X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # Load models
    print("\n" + "=" * 60)
    print("LOADING EXACT MODELS")
    print("=" * 60)
    
    models = load_models()
    model_names = ["CatBoost", "XGBoost", "HistGradientBoosting"]
    
    # Evaluate individual models on validation set
    print("\n" + "=" * 60)
    print("EVALUATING INDIVIDUAL MODELS ON VALIDATION SET")
    print("=" * 60)
    
    val_preds = []
    for model, name in zip(models, model_names):
        y_pred, _, _ = evaluate_model(model, X_val, y_val, model_name=name)
        val_preds.append(y_pred)
    
    # Evaluate ensemble on validation set
    print("\n" + "=" * 60)
    print("EVALUATING ENSEMBLE ON VALIDATION SET")
    print("=" * 60)
    
    _ = evaluate_ensemble(val_preds, y_val, TARGET_COLS)
    
    # Evaluate on full training set for thoroughness
    print("\n" + "=" * 60)
    print("EVALUATING ON FULL TRAINING SET (FOR REFERENCE)")
    print("=" * 60)
    
    train_preds = []
    for model, name in zip(models, model_names):
        y_pred, _, _ = evaluate_model(model, X_train_full, y_train_full, model_name=name)
        train_preds.append(y_pred)
    
    _ = evaluate_ensemble(train_preds, y_train_full, TARGET_COLS)
    
    # Generate submission file
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION FILE")
    print("=" * 60)
    
    _ = generate_submission(models, X_test_full, test_pid, TARGET_COLS)
    
    print("\nAll done! This submission should exactly match the original notebook's results.")

if __name__ == "__main__":
    main()
