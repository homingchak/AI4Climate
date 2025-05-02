import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from utils import load_and_clean_data, calculate_distance, calculate_wind_speed_change

def main():
    # df = load_and_clean_data('atlantic.csv')
    df = load_and_clean_data('pacific.csv')
    
    print(f"Total Records: {len(df)}")
    print(f"Total Unique Storms: {len(df['ID'].unique())}")
    
    hurricanes = df[df['Status'] == 'HU']
    print(f"Total Hurricanes: {len(hurricanes['ID'].unique())}")
    
    df = calculate_wind_speed_change(df)
    
    total_records = len(df)
    ri_events = df['Rapid_Intensification'].sum()
    print(f"RI Events: {ri_events} ({ri_events/total_records*100:.2f}% of records)")
    
    # Add storm speed feature
    df['Distance_km'] = df.groupby('ID').apply(
        lambda x: pd.Series([np.nan] + [calculate_distance(x['Latitude'].iloc[i], x['Longitude'].iloc[i], 
                                                           x['Latitude'].iloc[i+1], x['Longitude'].iloc[i+1]) 
                                         for i in range(len(x)-1)], index=x.index)
    ).reset_index(level=0, drop=True)
    df['Storm_Speed_km_h'] = df['Distance_km'] / df['Time_Diff_Hours']
    
    # Add time-based features
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    
    # One-hot encode the Event column
    event_dummies = pd.get_dummies(df['Event'], prefix='Event')
    df = pd.concat([df, event_dummies], axis=1)
    
    # Prepare features and target
    wind_radii_cols = ['Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW',
                       'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW',
                       'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
    features = ['Maximum Wind', 'Minimum Pressure', 'Latitude', 'Longitude', 
                'Storm_Speed_km_h', 'Year', 'Month'] + [col for col in df.columns if col.startswith('Event_')] + wind_radii_cols
    X = df[features].dropna()
    y = df.loc[X.index, 'Rapid_Intensification']
    
    print(f"Records after dropping NaN values in features: {len(X)}")
    
    # Stratified train-test split to maintain RI proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"RI proportion in training set: {y_train.mean():.3f}")
    print(f"RI proportion in test set: {y_test.mean():.3f}")
    
    # Define and tune XGBoost model
    model = XGBClassifier(scale_pos_weight=5, random_state=42, eval_metric='logloss'
    )
    '''
    # Atlantic
    param_grid = {
        'max_depth': [2, 3, 4],
        'min_child_weight': [0.5, 0.75, 1],
        'n_estimators': [20, 25, 30],
        'learning_rate': [0.25, 0.3, 0.35],
        'gamma': [0.15, 0.2, 0.25]
    }
    '''
    #'''
    # Pacific
    param_grid = {
        'max_depth': [2, 3, 4],
        'min_child_weight': [0.4, 0.5, 0.6],
        'n_estimators': [30, 35, 40],
        'learning_rate': [0.3, 0.35, 0.4],
        'gamma': [0, 0.02, 0.04]
    }
    #'''

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
    
    feature_groups = {}
    for feature in X.columns:
        # Group wind radii features as 'Low', 'Moderate', 'High'
        if 'Wind' in feature and feature != 'Maximum Wind' and any(prefix in feature for prefix in ['Low', 'Moderate', 'High']):
            group = feature.split(' ')[0] + ' Wind'
            feature_groups[feature] = group
        else:
            feature_groups[feature] = feature  

    grouped_importances = {}
    for feature, importance in zip(X.columns, model.feature_importances_):
        group = feature_groups[feature]
        grouped_importances[group] = grouped_importances.get(group, 0) + importance

    group_names = list(grouped_importances.keys())
    group_importances = list(grouped_importances.values())
    indices = np.argsort(group_importances)[::-1]
    sorted_groups = [group_names[i] for i in indices]
    sorted_group_importances = [group_importances[i] for i in indices]

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_groups, sorted_group_importances, color='skyblue')
    plt.title('Feature Importance for XGBoost')
    plt.xlabel('Feature Groups')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance_xgboost.png')
    plt.close()
    print("Grouped feature importance plot saved as 'feature_importance_xgboost.png'")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
