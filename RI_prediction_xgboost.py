import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve
from xgboost import XGBClassifier
from utils import load_and_clean_data, calculate_distance

def calculate_wind_speed_change(df):
    df = df.sort_values(['ID', 'Datetime'])
    
    # Calculate time difference (in hours) between consecutive records for each storm
    df['Time_Diff_Hours'] = df.groupby('ID')['Datetime'].diff().dt.total_seconds() / 3600
    
    # Calculate wind speed change over 24 hours
    df['Wind_Speed_Diff'] = df.groupby('ID')['Maximum Wind'].diff()
    df['Wind_Speed_Change_Rate'] = df['Wind_Speed_Diff'] / df['Time_Diff_Hours']
    df['Wind_Speed_Change_24h'] = df['Wind_Speed_Change_Rate'] * 24
    
    # Label as Rapid Intensification (RI) if wind speed increases by >= 30 knots in 24 hours
    df['Rapid_Intensification'] = (df['Wind_Speed_Change_24h'] >= 30).astype(int)
    
    return df

def main():
    df = load_and_clean_data('atlantic.csv')
    
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
    model = XGBClassifier(
        scale_pos_weight=5,
        random_state=42, eval_metric='logloss'
    )
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    
    model = grid_search.best_estimator_
    
    # Cross-validation with best model
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"Cross-Validation F1 Macro Scores: {cv_scores}")
    print(f"Average CV F1 Macro Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Predict probabilities and find optimal threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # Apply threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    print("Classification Report on Test Set (with optimal threshold):")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()