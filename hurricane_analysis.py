import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import load_and_clean_data

def plot_hurricane_tracks_on_map(df, min_wind=100):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-100, -10, 0, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    
    strong_hurricanes = df[(df['Status'] == 'HU') & (df['Maximum Wind'] >= min_wind)]
    print(f"Number of records with Status == 'HU' and Maximum Wind >= {min_wind}: {len(strong_hurricanes)}")
    print(f"Number of storms with Status == 'HU' and Maximum Wind >= {min_wind}: {len(strong_hurricanes['ID'].unique())}")
    
    num_storms = len(strong_hurricanes['ID'].unique())
    if num_storms == 0:
        print(f"Warning: No storms meet the criteria for plotting tracks (Status == 'HU' and Maximum Wind >= {min_wind}).")
        plt.title(f"Tracks of Hurricanes with Maximum Wind ≥ {min_wind} knots on Map (No Data)")
        plt.savefig('hurricane_tracks_map.png')
        plt.close()
        return
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_storms, 20)))
    color_cycle = np.repeat(colors, np.ceil(num_storms / 20).astype(int), axis=0)[:num_storms]
    
    for idx, storm_id in enumerate(strong_hurricanes['ID'].unique()):
        storm_data = strong_hurricanes[strong_hurricanes['ID'] == storm_id]
        storm_data = storm_data.sort_values('Datetime')
        
        if len(storm_data) < 2:
            print(f"Storm {storm_id}: Not enough points to plot (need at least 2 points).")
            continue
        
        ax.plot(storm_data['Longitude'], storm_data['Latitude'],
                color=color_cycle[idx], linewidth=2, alpha=0.7, transform=ccrs.PlateCarree())
    
    plt.title(f'Tracks of Hurricanes with Maximum Wind ≥ {min_wind} knots on Map')
    plt.savefig('hurricane_tracks_map.png')
    plt.close()

def plot_wind_speed_distribution(df):
    plt.figure(figsize=(10, 6))
    hurricanes = df[df['Status'] == 'HU']
    wind_speeds = hurricanes['Maximum Wind'].dropna()
    
    min_speed = int(np.floor(wind_speeds.min() / 5) * 5)
    max_speed = int(np.ceil(wind_speeds.max() / 5) * 5)
    bins = np.arange(min_speed, max_speed + 5, 5)
    
    plt.hist(wind_speeds, bins=bins, edgecolor='black', color='skyblue', align='mid')
    plt.title('Distribution of Maximum Wind Speeds for Hurricanes')
    plt.xlabel('Maximum Wind Speed (knots)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('wind_speed_distribution.png')
    plt.close()

def plot_hurricane_frequency(df):
    plt.figure(figsize=(12, 6))
    hurricanes = df[df['Status'] == 'HU'][['ID', 'Year']].drop_duplicates()
    yearly_counts = hurricanes.groupby('Year').size()
    
    years = yearly_counts.index
    counts = yearly_counts.values
    
    plt.plot(years, counts, marker='o', color='coral', label='Hurricane Frequency', linestyle='-')
    coefficients = np.polyfit(years, counts, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(years, trendline(years), color='blue', linestyle='--', label=f'Trendline (slope={coefficients[0]:.4f})')
    
    plt.title('Unique Hurricanes per Year with Trendline')
    plt.xlabel('Year')
    plt.ylabel('Number of Hurricanes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('hurricane_frequency.png')
    plt.close()

def main():
    data = load_and_clean_data('atlantic.csv')
    data['Year'] = data['Datetime'].dt.year
    
    print(f"Records with missing Maximum Wind: {data['Maximum Wind'].isna().sum()}")
    wind_radii_cols = ['Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW',
                       'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW',
                       'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
    for col in wind_radii_cols:
        print(f"Records with missing {col}: {data[col].isna().sum()}")
    
    plot_hurricane_tracks_on_map(data, min_wind=100)
    plot_wind_speed_distribution(data)
    plot_hurricane_frequency(data)
    
    hurricanes = data[data['Status'] == 'HU']
    avg_wind = hurricanes['Maximum Wind'].mean()
    print(f"Total Hurricanes: {len(hurricanes['ID'].unique())}")
    print(f"Average Maximum Wind Speed: {avg_wind:.2f} knots")
    print(f"Years Covered: {data['Year'].min()} to {data['Year'].max()}")
    print("Map plot saved as 'hurricane_tracks_map.png'")
    print("Wind speed distribution saved as 'wind_speed_distribution.png'")
    print("Hurricane frequency plot saved as 'hurricane_frequency.png'")

if __name__ == "__main__":
    main()