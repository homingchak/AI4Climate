import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import load_and_clean_data_hko, calculate_distance, calculate_wind_speed_change_hko

def main():
    df = load_and_clean_data_hko('HKO_BST.csv')
    
    print(f"Total Records: {len(df)}")
    print(f"Total Unique Storms: {len(df['Typhoon_ID'].unique())}")
    
    typhoons = df[df['Intensity'].isin(['T', 'ST', 'SuperT'])]
    print(f"Total Typhoons: {len(typhoons['Typhoon_ID'].unique())}")
    
    df = calculate_wind_speed_change_hko(df)
    
    total_records = len(df)
    ri_events = df['Rapid_Intensification'].sum()
    print(f"RI Events: {ri_events} ({ri_events/total_records*100:.2f}% of records)")
    
    # Add storm speed feature
    df['Distance_km'] = df.groupby('Typhoon_ID').apply(
        lambda x: pd.Series([np.nan] + [calculate_distance(x['Latitude'].iloc[i], x['Longitude'].iloc[i], 
                                                           x['Latitude'].iloc[i+1], x['Longitude'].iloc[i+1]) 
                                         for i in range(len(x)-1)], index=x.index)
    ).reset_index(level=0, drop=True)
    
    df['Storm_Speed_km_h'] = df['Distance_km'] / df['Time_Diff_Hours']
    
    # Add time-based features
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    
    # Prepare features and target
    features = ['Estimated maximum surface winds (knot)', 'Estimated minimum central pressure (hPa)', 
                'Latitude', 'Longitude', 'Storm_Speed_km_h', 'Year', 'Month']
    X = df[features].dropna()
    y = df.loc[X.index, 'Rapid_Intensification']
    
    print(f"Records after dropping NaN values in features: {len(X)}")
    
    # Stratified train-test split to maintain RI proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"RI proportion in training set: {y_train.mean():.3f}")
    print(f"RI proportion in test set: {y_test.mean():.3f}")
    
    # Define and tune model with grid search
    param_grid = {
        'max_depth': [15, 20, 25],
        'min_samples_split': [20, 25, 30],
        'min_samples_leaf': [3, 4, 5],
        'n_estimators': [125, 150, 175]
    }

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro', n_jobs=-1
    )
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    minutes, seconds = divmod(training_time, 60)
    print(f"Grid Search Training Time: {training_time:.2f} seconds ({int(minutes)} minutes, {seconds:.2f} seconds)")

    print("Best Parameters:", grid_search.best_params_)
    
    model = grid_search.best_estimator_
    
    # Cross-validation with best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Cross-Validation Macro F1 Scores: {cv_scores}")
    print(f"Average CV Macro F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_features, sorted_importances, color='skyblue')
    plt.title('Feature Importance for Random Forest HKO')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig('hko_feature_importance_random_forest.png')
    plt.close()
    print("Feature importance plot saved as 'hko_feature_importance_random_forest.png'")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
