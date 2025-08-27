# ASIA Motor Score Prediction Challenge - Winning Solution

## Project Overview
This repository contains the code for my winning solution to the ASIA Challenge Kaggle competition, which focused on predicting patient motor score impairment at 26 weeks and 52 weeks based on week 1 data for spinal cord injury patients.

## Competition Goal
The objective was to develop a machine learning model that accurately predicts 20 different motor scores for patients with spinal cord injuries at the 26-week and 52-week marks, using only data available from the first week post-injury. These predictions help clinicians understand potential recovery trajectories early in the treatment process.

## Target Variables
The model predicts 20 motor score outcomes represented by the following target variables:
- `elbfll`, `wrextl`, `elbexl`, `finfll`, `finabl`, `hipfll`, `kneexl`, `ankdol`, `gretol`, `ankpll` (left side)
- `elbflr`, `wrextr`, `elbexr`, `finflr`, `finabr`, `hipflr`, `kneetr`, `ankdor`, `gretor`, `ankplr` (right side)

These variables represent different muscle group functions assessed on a scale of 0-5.

## Methodology

### Data Processing
- Starting with preprocessed training and test data
- Feature engineering and data cleaning performed in earlier notebooks
- Separation of features and target variables

### Ensemble Approach
My winning solution leverages an ensemble of three gradient boosting algorithms:
1. **CatBoost**
2. **XGBoost**
3. **HistGradientBoosting** (from scikit-learn)

Each algorithm was implemented using `MultiOutputRegressor` to handle the multiple target variables simultaneously.

### Hyperparameter Optimization
- Hyperparameters were optimized using a rigorous cross-validation strategy (detailed in notebook 02)
- Different parameter spaces were explored for each algorithm
- Final parameters were selected based on validation performance

### Model Training & Prediction Pipeline
1. Train each model on the full training dataset
2. Generate predictions from each model
3. Create an ensemble prediction by averaging the outputs
4. Round and clip predictions to ensure they fall within the valid range (0-5)

### Training Performance
- **Average RMSE across 20 targets**: 0.8909
- **Average MAE across 20 targets**: 0.4655

## Repository Structure

### Notebooks (Original Development)
- `notebooks/`
  - `01_data_preprocessing.ipynb`: Initial data loading, cleaning, and feature processing
  - `02_model_building_and_hpo.ipynb`: Model selection and hyperparameter optimization
  - `03_ensemble_final_training_and_submission.ipynb`: Final ensemble model training and submission generation

### Reproduction Scripts
- `end_to_end_reproduction.py`: **Main reproduction script** that loads preprocessed data, trained models, and generates predictions
- `exact_replication.py`: Script to train and save models with the same parameters as the original notebook
- `load_exact_models.py`: Script to load saved models and generate predictions without retraining

### Models and Data
- `models_exact/`: Contains the saved model files
  - `catboost_exact_model.pkl`: Trained CatBoost ensemble model
  - `xgb_exact_model.pkl`: Trained XGBoost ensemble model
  - `hgb_exact_model.pkl`: Trained HistGradientBoosting ensemble model
- `data/`: Contains processed data files and model parameters
  - `preprocessing_pipeline.pkl`: Saved preprocessing pipeline for data transformation
  - `best_params_catboost.pkl`: Optimized hyperparameters for CatBoost model
  - `best_params_xgb.pkl`: Optimized hyperparameters for XGBoost model
  - `best_params_hgb.pkl`: Optimized hyperparameters for HistGradientBoosting model
- `reproduced_results/`: Contains generated submission files

## Reproducibility

This repository has been carefully structured to ensure reproducibility of the competition results. The end-to-end reproduction starts with the raw data files, which provides full transparency into the entire pipeline from data processing to prediction.

Starting from raw data files allows anyone to understand the entire process and verify each step independently.

## Requirements

### Python Version
This project was developed and tested using Python 3.12.3. It's recommended to use the same version to ensure full compatibility.

### Libraries

All required dependencies are listed in the `requirements.txt` file. You can install them with:

```
pip install -r requirements.txt
```

### Data Setup
**Important for Contest Reviewers**: To run the end-to-end reproduction script, you need to place the original competition data files in the `data/` directory:

1. From the `ASIAChallenge_ShareFile` provided in the competition, copy these files to the `data/` folder:
   - `train_features.csv`
   - `train_outcomes_ms.csv`
   - `test_features.csv`
   - `metadata.csv`

These files are required for running the script, as it processes the raw data files by default.

## How to Reproduce Results

### Option 1: End-to-End Reproduction (Recommended)
```bash
python end_to_end_reproduction.py
```
This script:
1. Processes data from raw files (default behavior)
2. Loads the trained models
3. Performs validation on training data
4. Generates predictions on test data
5. Creates a submission file

Command-line options:
- Default (no arguments): Processes from raw data files
- `--use-saved-pipeline`: Uses saved preprocessing files if available

### Option 2: Using Pre-trained Models Only
```bash
python load_exact_models.py
```
This script loads the saved models and generates predictions without retraining.

### Option 3: Complete Retraining
```bash
python exact_replication.py
```
This script retrains the models from scratch with the exact same parameters as in the notebooks. It uses the saved best hyperparameters to ensure the models are trained identically.

## Acknowledgments
I would like to thank the competition organizers for providing this valuable dataset and the opportunity to work on such an impactful problem. The ability to predict motor outcomes in spinal cord injury patients can significantly improve treatment planning and patient care.

---

Created by Jesse Villines, April 2025
