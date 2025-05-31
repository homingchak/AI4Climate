# AI4Climate: Machine Learning for Weather System Identification and Classification

This project leverages machine learning techniques, specifically Random Forest classifiers and XGBoost, to predict rapid intensification (RI) of tropical cyclones using datasets from the National Oceanic and Atmospheric Administration (NOAA) and the Hong Kong Observatory (HKO). The goal is to enhance weather system analysis and classification for improved forecasting and research.

## Overview

- **Datasets**:
  - NOAA Hurricane Database: Includes `atlantic.csv` and `pacific.csv`, available at [https://www.kaggle.com/datasets/noaa/hurricane-database/data](https://www.kaggle.com/datasets/noaa/hurricane-database/data).
  - HKO Tropical cyclone best track data: Merged into one single file `HKO_BST.csv`, available at [https://portal.csdi.gov.hk/geoportal/?lang=en&datasetId=b7b4acbe-adb8-bcac-bbb6-af908d8b9e93](https://portal.csdi.gov.hk/geoportal/?lang=en&datasetId=b7b4acbe-adb8-bcac-bbb6-af908d8b9e93).

- **Objective**: Explore and compare machine learning models to predict rapid intensification of tropical cyclones across multiple datasets.

- **Utilities**: The `utils.py` file contains shared functions and does not need to be run independently.

## Repository Structure

| File/Directory          | Description                                      |
|--------------------------|--------------------------------------------------|
| `utils.py`              | Utility functions for data loading and processing. |
| `merge_hko.py`              | Merges HKO best track data of each year into one single file. |
| `hurricane_analysis.py` | Analyzes Atlantic and Pacific datasets, generating summaries, maps of hurricanes, wind speed histograms, and hurricane frequency graphs. |
| `typhoon_analysis_hko.py` | Analyzes HKO dataset, generating summary, map of typhoons, wind speed histogram, and typhoon frequency graph. |
| `RI_prediction_random_forest_v1.py` | Trains a Random Forest model (V1) with 4 features, evaluates it, and plots feature importance. |
| `RI_prediction_random_forest_v2.py` | Trains a Random Forest model (V2) with 8 features, evaluates it, and plots feature importance. |
| `RI_prediction_random_forest_v3.py` | Trains a Random Forest model (V3) with all features, evaluates it, and plots feature importance. |
| `RI_prediction_xgboost.py` | Trains an XGBoost model with all features, evaluates it, and plots feature importance. |
| `RI_prediction_random_forest_hko.py` | Trains a Random Forest model on the HKO dataset, evaluates it, and plots feature importance. |
| `RI_prediction_xgboost_hko.py` | Trains an XGBoost model on the HKO dataset, evaluates it, and plots feature importance. |

## Setup and Usage

1. **Prerequisites**:
   - Ensure Python is installed with required libraries: `pandas`, `numpy`, `matplotlib`, `cartopy`, `geopy`, `scikit-learn`, `xgboost`.

2. **Data Preparation**:
   - Place `atlantic.csv`, `pacific.csv`, and `HKO_BST.csv` in the project directory.

3. **Running Scripts**:
   - **For NOAA Datasets (`atlantic.csv` or `pacific.csv`)**:
     - Run `hurricane_analysis.py` to generate summaries, maps of hurricanes, wind speed histograms, and hurricane frequency graphs for both datasets..
     - Open `RI_prediction_random_forest_v1.py`, `RI_prediction_random_forest_v2.py`, `RI_prediction_random_forest_v3.py`, or `RI_prediction_xgboost.py`.
     - In the `main()` function, uncomment the desired dataset line (e.g., `df = load_and_clean_data('atlantic.csv')`) and the corresponding `param_grid` section optimized for that dataset.
     - Run the script to train the model, generate a summary, classification report, and feature importance plot.
   - **For HKO Dataset (`HKO_BST.csv`)**:
     - Run `typhoon_analysis_hko.py` to generate summary, map of typhoons, wind speed histogram, and typhoon frequency graph for the HKO dataset.
     - Run `RI_prediction_random_forest_hko.py` or `RI_prediction_xgboost_hko.py` directly to train the model, generate a summary, classification report, and feature importance plot.

## Output

- Each script outputs:
  - A summary of dataset statistics (e.g., total records, RI events).
  - Training time and best hyperparameters.
  - Cross-validation F1 scores and a classification report (precision, recall, F1-score).
  - A feature importance plot saved as a `.png` file.
