import pandas as pd
import numpy as np
from datetime import datetime
import geopy.distance

def parse_lat_lon(coord):
    if not isinstance(coord, str):
        return np.nan
    try:
        value = float(coord[:-1])
        direction = coord[-1]
        if direction in ['S', 'W']:
            value = -value
        return value
    except (ValueError, TypeError):
        return np.nan

def parse_datetime(row):
    try:
        date_str = str(row['Date'])
        time_str = str(row['Time']).zfill(4)
        return datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
    except (ValueError, TypeError):
        return pd.NaT

def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geopy.distance.distance(coords_1, coords_2).km
    except:
        return np.nan
    
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

def load_and_clean_data(file_path, filter_hurricanes=False):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    
    numeric_cols = ['Maximum Wind', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE', 
                    'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE', 'Moderate Wind SE',
                    'Moderate Wind SW', 'Moderate Wind NW', 'High Wind NE', 'High Wind SE',
                    'High Wind SW', 'High Wind NW']
    df[numeric_cols] = df[numeric_cols].replace(-999, np.nan)
    df[numeric_cols] = df[numeric_cols].replace(-99, np.nan)


    df['Latitude'] = df['Latitude'].apply(parse_lat_lon)
    df['Longitude'] = df['Longitude'].apply(parse_lat_lon)
    
    df['Datetime'] = df.apply(parse_datetime, axis=1)
    
    df = df.dropna(subset=['Datetime', 'Latitude', 'Longitude'])
    
    if filter_hurricanes:
        df = df[df['Status'] == 'HU']
    
    return df

def calculate_wind_speed_change_hko(df):
    df = df.sort_values(['Typhoon_ID', 'Datetime'])
    
    # Calculate time difference (in hours) between consecutive records for each typhoon
    df['Time_Diff_Hours'] = df.groupby('Typhoon_ID')['Datetime'].diff().dt.total_seconds() / 3600
    
    # Calculate wind speed change over 24 hours
    df['Wind_Speed_Diff'] = df.groupby('Typhoon_ID')['Estimated maximum surface winds (knot)'].diff()
    df['Wind_Speed_Change_Rate'] = df['Wind_Speed_Diff'] / df['Time_Diff_Hours']
    df['Wind_Speed_Change_24h'] = df['Wind_Speed_Change_Rate'] * 24
    
    # Label as Rapid Intensification (RI) if wind speed increases by >= 30 knots in 24 hours
    df['Rapid_Intensification'] = (df['Wind_Speed_Change_24h'] >= 30).astype(int)
    
    return df

def load_and_clean_data_hko(file_path, filter_typhoons=False):
    df = pd.read_csv(file_path)

    df['Month'] = df['Month'].astype(str).str.zfill(2)
    df['Day'] = df['Day'].astype(str).str.zfill(2)
    df['Time (UTC)'] = df['Time (UTC)'].astype(str).str.zfill(2)
    
    df['Datetime'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                                   df['Month'] + '-' + 
                                   df['Day'] + ' ' + 
                                   df['Time (UTC)'] + ':00', 
                                   format='%Y-%m-%d %H:%M')
    
    df['Latitude'] = df['Latitude (0.01 degree N)'] / 100
    df['Longitude'] = df['Longitude (0.01 degree E)'] / 100
    
    df['Typhoon_ID'] = df['Tropical Cyclone Name'].astype(str) + '_' + df['Year'].astype(str) + '_' + df['HKO Code'].astype(str)
    
    if filter_typhoons:
        typhoon_statuses = ['T', 'ST', 'SuperT']
        df = df[df['Intensity'].isin(typhoon_statuses)]

    return df
