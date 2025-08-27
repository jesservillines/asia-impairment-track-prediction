#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-End Reproduction Script

This script performs the complete workflow:
1) Loads raw data files
2) Performs preprocessing using the same pipeline as the original notebook
3) Loads the trained exact models
4) Performs inference and validation
5) Generates a submission file that exactly matches the original results

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
import time
from datetime import datetime
from pathlib import Path

# For preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# For evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =========================
# 2. Define File Paths
# =========================

# Directory structure
DATA_DIR = "./data"
MODELS_DIR = "./models_exact"
OUTPUT_DIR = "./reproduced_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Raw input files
TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, "train_features.csv")
TRAIN_OUTCOMES_PATH = os.path.join(DATA_DIR, "train_outcomes_ms.csv")
TEST_FEATURES_PATH  = os.path.join(DATA_DIR, "test_features.csv")
METADATA_PATH       = os.path.join(DATA_DIR, "metadata.csv")

# Intermediate files
PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, "train_processed.csv")
PROCESSED_TEST_PATH  = os.path.join(DATA_DIR, "test_processed.csv")
PIPELINE_PATH        = os.path.join(DATA_DIR, "preprocessing_pipeline.pkl")

# Model files
CATBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "catboost_exact_model.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_exact_model.pkl")
HGB_MODEL_PATH = os.path.join(MODELS_DIR, "hgb_exact_model.pkl")

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, f"submission_end_to_end_{TIMESTAMP}.csv")

# =========================
# 3. Preprocessing Functions
# =========================

def get_feature_names_from_column_transformer(ct, numeric_cols, categorical_cols):
    """
    Utility to extract the final feature names from a ColumnTransformer with numeric and cat transformers.
    """
    # For numeric columns, they keep the same name
    num_features = numeric_cols

    # For categorical columns, the OneHotEncoder creates new features
    cat_pipeline = ct.named_transformers_['cat']
    cat_encoder = cat_pipeline.named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)

    return list(num_features) + list(cat_feature_names)

def preprocessing_pipeline(use_saved_pipeline=True):
    """
    Process data for model prediction
    
    Parameters:
    -----------
    use_saved_pipeline : bool, default=True
        If True, use the saved preprocessing pipeline for 100% exact reproduction
        If False, reprocess from raw files (may lead to slight differences from floating point precision)
    """
    print("=" * 60)
    print("PREPROCESSING - LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Check if we should use the saved pipeline or reprocess from raw files
    if use_saved_pipeline:
        print("Using saved preprocessing pipeline for 100% exact reproduction...")
        
        # Check if processed files exist
        if os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH):
            print("Loading existing processed data files...")
            train_processed_df = pd.read_csv(PROCESSED_TRAIN_PATH)
            test_processed_df = pd.read_csv(PROCESSED_TEST_PATH)
            
            print(f"Loaded processed train shape: {train_processed_df.shape}")
            print(f"Loaded processed test shape: {test_processed_df.shape}")
            
            return train_processed_df, test_processed_df, None
        else:
            raise FileNotFoundError(
                "Processed data files not found but use_saved_pipeline=True.\n"
                f"Expected files:\n{PROCESSED_TRAIN_PATH}\n{PROCESSED_TEST_PATH}"
            )
    else:
        print("Processing data from raw files (may lead to slight differences)...")
    
    # Run the entire preprocessing pipeline
    
    # 4. Load Raw Data
    print("Loading raw data files...")
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    train_outcomes = pd.read_csv(TRAIN_OUTCOMES_PATH)
    test_features  = pd.read_csv(TEST_FEATURES_PATH)
    metadata       = pd.read_csv(METADATA_PATH)
    
    print(f"Shapes:")
    print(f"  train_features: {train_features.shape}")
    print(f"  train_outcomes: {train_outcomes.shape}")
    print(f"  test_features : {test_features.shape}")
    print(f"  metadata      : {metadata.shape}")
    
    # 5. Merge Train and Test with Metadata
    print("\nMerging datasets...")
    train_merged = pd.merge(train_features, metadata, on="PID", how="left")
    train_merged = pd.merge(train_merged, train_outcomes, on="PID", how="inner")
    test_merged = pd.merge(test_features, metadata, on="PID", how="left")
    
    print(f"After merging train features + metadata + outcomes: {train_merged.shape}")
    print(f"After merging test features + metadata: {test_merged.shape}")
    
    # 7. Identify Target Columns
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Ensure all target columns are in train_merged
    missing_targets = [col for col in TARGET_COLS if col not in train_merged.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in train_merged: {missing_targets}")
    
    # 8. Identify Common Features in Train & Test
    train_cols_set = set(train_merged.columns)
    test_cols_set = set(test_merged.columns)
    common_cols = list(train_cols_set.intersection(test_cols_set))
    
    # Remove target columns from the common columns
    common_feature_cols = [col for col in common_cols if col not in TARGET_COLS and col != "PID"]
    common_feature_cols = sorted(common_feature_cols)  # for consistent ordering
    
    print(f"\nNumber of common feature columns: {len(common_feature_cols)}")
    print(f"Some common feature columns: {common_feature_cols[:10]}")
    
    # 9. Split Data into Features and Targets
    X_train_full = train_merged[["PID"] + common_feature_cols].copy()
    y_train_full = train_merged[TARGET_COLS].copy()
    X_test_full = test_merged[["PID"] + common_feature_cols].copy()
    
    print(f"\nShapes after alignment:")
    print(f"  X_train_full: {X_train_full.shape}")
    print(f"  y_train_full: {y_train_full.shape}")
    print(f"  X_test_full : {X_test_full.shape}")
    
    # 10. Identify Numeric and Categorical Columns
    numeric_cols = []
    categorical_cols = []
    
    for col in common_feature_cols:
        if pd.api.types.is_numeric_dtype(X_train_full[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Build Preprocessing Pipeline
    # Numeric transformer with IterativeImputer
    numeric_transformer = IterativeImputer(
        max_iter=20,  # Increased from 10 to ensure convergence
        initial_strategy='mean',
        imputation_order='ascending',
        random_state=42
    )
    
    # Categorical transformer with SimpleImputer + OneHot
    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Build the final pipeline
    preproc_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    # 11. Fit Transform on Train, Transform Test
    train_pid = X_train_full['PID'].values
    X_train_no_pid = X_train_full.drop(columns=['PID'])
    
    test_pid = X_test_full['PID'].values
    X_test_no_pid = X_test_full.drop(columns=['PID'])
    
    # Fit the pipeline on train (features only)
    print("\nFitting the preprocessing pipeline on the training data...")
    preproc_pipeline.fit(X_train_no_pid)
    
    print("Transforming the training data...")
    X_train_transformed = preproc_pipeline.transform(X_train_no_pid)
    print("Transforming the test data...")
    X_test_transformed = preproc_pipeline.transform(X_test_no_pid)
    
    print(f"\nShapes after transformation:")
    print(f"  X_train_transformed: {X_train_transformed.shape}")
    print(f"  X_test_transformed : {X_test_transformed.shape}")
    
    # 12. Reconstruct Processed DataFrames
    final_feature_names = get_feature_names_from_column_transformer(
        preprocessor, numeric_cols, categorical_cols
    )
    
    # Construct processed train DataFrame (features only)
    X_train_df = pd.DataFrame(
        data=X_train_transformed,
        columns=final_feature_names,
    )
    X_train_df["PID"] = train_pid
    
    # Attach targets to this DataFrame for convenience
    train_processed_df = pd.concat([X_train_df, y_train_full.reset_index(drop=True)], axis=1)
    
    # Construct processed test DataFrame
    test_processed_df = pd.DataFrame(
        data=X_test_transformed,
        columns=final_feature_names
    )
    test_processed_df["PID"] = test_pid
    
    print(f"\nFinal train_processed_df shape: {train_processed_df.shape}")
    print(f"Final test_processed_df shape : {test_processed_df.shape}")
    
    # 13. Save Processed Data
    train_processed_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_processed_df.to_csv(PROCESSED_TEST_PATH, index=False)
    
    print(f"\nSaved processed train to: {PROCESSED_TRAIN_PATH}")
    print(f"Saved processed test  to: {PROCESSED_TEST_PATH}")
    
    # Save the pipeline for re-use
    joblib.dump(preproc_pipeline, PIPELINE_PATH)
    print(f"Preprocessing pipeline saved to: {PIPELINE_PATH}")
    
    return train_processed_df, test_processed_df, preproc_pipeline

# =========================
# 4. Model Loading & Evaluation Functions
# =========================

def load_models():
    """Load all models from disk"""
    print("\n" + "=" * 60)
    print("LOADING EXACT MODELS")
    print("=" * 60)
    
    # Check if models exist
    if not all(os.path.exists(path) for path in [CATBOOST_MODEL_PATH, XGB_MODEL_PATH, HGB_MODEL_PATH]):
        raise FileNotFoundError(
            "One or more model files not found. Please run exact_replication.py first."
            f"\nExpected model paths:\n{CATBOOST_MODEL_PATH}\n{XGB_MODEL_PATH}\n{HGB_MODEL_PATH}"
        )
    
    # Load models
    print("Loading models from disk...")
    catboost_multi = joblib.load(CATBOOST_MODEL_PATH)
    print(f"  Loaded CatBoost model from {CATBOOST_MODEL_PATH}")
    
    xgb_multi = joblib.load(XGB_MODEL_PATH)
    print(f"  Loaded XGBoost model from {XGB_MODEL_PATH}")
    
    hgb_multi = joblib.load(HGB_MODEL_PATH)
    print(f"  Loaded HistGradientBoosting model from {HGB_MODEL_PATH}")
    
    print("All models loaded successfully!")
    
    return catboost_multi, xgb_multi, hgb_multi

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
    
    return ensemble_pred_clipped

def generate_submission(models, X_test, test_pid, target_cols, output_path=SUBMISSION_PATH):
    """Generate predictions using loaded models and save submission file - exactly as in original"""
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION FILE")
    print("=" * 60)
    
    catboost_multi, xgb_multi, hgb_multi = models
    
    print("Generating predictions for test set...")
    
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

# Removed comparison function

# =========================
# 5. Main Function
# =========================

def main(use_saved_pipeline=True):
    """Main function to run the entire workflow"""
    start_time = time.time()
    
    print("=" * 60)
    print("STARTING END-TO-END REPRODUCTION WORKFLOW")
    print("=" * 60)
    
    # Step 1: Preprocessing
    train_df, test_df, _ = preprocessing_pipeline(use_saved_pipeline=use_saved_pipeline)
    
    # Extract features and targets
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Store test PIDs
    test_pid = test_df['PID'].values
    
    # Remove PID from features if present
    if 'PID' in train_df.columns:
        train_df_no_pid = train_df.drop(columns=['PID'])
    else:
        train_df_no_pid = train_df
    
    if 'PID' in test_df.columns:
        test_df_no_pid = test_df.drop(columns=['PID'])
    else:
        test_df_no_pid = test_df
    
    # Separate features and targets
    X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
    X_train_full = train_df_no_pid[X_cols].values
    y_train_full = train_df_no_pid[TARGET_COLS].values if all(col in train_df_no_pid.columns for col in TARGET_COLS) else None
    X_test_full = test_df_no_pid[X_cols].values
    
    # Step 2: Load Models
    models = load_models()
    model_names = ["CatBoost", "XGBoost", "HistGradientBoosting"]
    
    # Step 3: Optional - Evaluate on training data
    if y_train_full is not None:
        print("\n" + "=" * 60)
        print("EVALUATING MODELS ON TRAINING DATA")
        print("=" * 60)
        
        train_preds = []
        for model, name in zip(models, model_names):
            y_pred, _, _ = evaluate_model(model, X_train_full, y_train_full, model_name=name)
            train_preds.append(y_pred)
        
        _ = evaluate_ensemble(train_preds, y_train_full, TARGET_COLS)
    
    # Step 4: Generate Submission
    submission_df = generate_submission(models, X_test_full, test_pid, TARGET_COLS)
    
    # Done!
    end_time = time.time()
    duration_mins = (end_time - start_time) / 60
    print("\n" + "=" * 60)
    print("END-TO-END REPRODUCTION WORKFLOW COMPLETE")
    print(f"Total execution time: {duration_mins:.2f} minutes")
    print("=" * 60)
    
    print(f"\nReproduction completed successfully!")
    print(f"Submission file saved to: {SUBMISSION_PATH}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="End-to-end reproduction of competition results")
    parser.add_argument(
        "--use-saved-pipeline", 
        dest="use_saved_pipeline",
        action="store_true", 
        default=False,
        help="Use saved preprocessing pipeline for 100% exact reproduction"
    )
    parser.add_argument(
        "--reprocess-raw-files", 
        dest="use_saved_pipeline",
        action="store_false",
        help="Reprocess data from raw files (default: True)"
    )
    
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    main(use_saved_pipeline=args.use_saved_pipeline)
