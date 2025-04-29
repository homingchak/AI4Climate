import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import load_and_clean_data

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
    
    # Prepare features and target
    features = ['Maximum Wind', 'Minimum Pressure', 'Latitude', 'Longitude']
    X = df[features].dropna()
    y = df.loc[X.index, 'Rapid_Intensification']
    
    print(f"Records after dropping NaN values in features: {len(X)}")
    
    # Stratified train-test split to maintain RI proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"RI proportion in training set: {y_train.mean():.3f}")
    print(f"RI proportion in test set: {y_test.mean():.3f}")
    
    # Define model
    model = RandomForestClassifier(
        max_depth=10, min_samples_split=10, random_state=42
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Cross-Validation F1 Macro Scores: {cv_scores}")
    print(f"Average CV F1 Macro Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()