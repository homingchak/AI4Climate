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