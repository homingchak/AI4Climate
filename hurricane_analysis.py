import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import load_and_clean_data

def plot_hurricane_tracks_on_map(df, min_wind=100, dataset_name="atlantic"):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    if dataset_name.lower() == "atlantic":
        ax.set_extent([-100, -10, 0, 60], crs=ccrs.PlateCarree())
    elif dataset_name.lower() == "pacific":
        ax.set_extent([-180, -100, 0, 60], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    
    strong_hurricanes = df[(df['Status'] == 'HU') & (df['Maximum Wind'] >= min_wind)]
    print(f"{dataset_name.capitalize()} Dataset: Number of records with Status == 'HU' and Maximum Wind >= {min_wind}: {len(strong_hurricanes)}")
    print(f"{dataset_name.capitalize()} Dataset: Number of storms with Status == 'HU' and Maximum Wind >= {min_wind}: {len(strong_hurricanes['ID'].unique())}")
    
    num_storms = len(strong_hurricanes['ID'].unique())
    if num_storms == 0:
        print(f"Warning: No storms meet the criteria for plotting tracks (Status == 'HU' and Maximum Wind >= {min_wind}) in {dataset_name.capitalize()} dataset.")
        plt.title(f"Tracks of {dataset_name.capitalize()} Hurricanes with Maximum Wind ≥ {min_wind} knots on Map (No Data)")
        plt.savefig(f'{dataset_name}_hurricane_tracks_map.png')
        plt.close()
        return
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_storms, 20)))
    color_cycle = np.repeat(colors, np.ceil(num_storms / 20).astype(int), axis=0)[:num_storms]
    
    for idx, storm_id in enumerate(strong_hurricanes['ID'].unique()):
        storm_data = strong_hurricanes[strong_hurricanes['ID'] == storm_id]
        storm_data = storm_data.sort_values('Datetime')
        
        if len(storm_data) < 2:
            print(f"{dataset_name.capitalize()} Dataset: Storm {storm_id}: Not enough points to plot (need at least 2 points).")
            continue
        
        ax.plot(storm_data['Longitude'], storm_data['Latitude'],
                color=color_cycle[idx], linewidth=2, alpha=0.7, transform=ccrs.PlateCarree())
    
    plt.title(f'Tracks of {dataset_name.capitalize()} Hurricanes with Maximum Wind ≥ {min_wind} knots on Map')
    plt.savefig(f'{dataset_name}_hurricane_tracks_map.png')
    plt.close()

def plot_wind_speed_distribution(df, dataset_name="atlantic"):
    plt.figure(figsize=(10, 6))
    hurricanes = df[df['Status'] == 'HU']
    wind_speeds = hurricanes['Maximum Wind'].dropna()
    
    min_speed = int(np.floor(wind_speeds.min() / 5) * 5)
    max_speed = int(np.ceil(wind_speeds.max() / 5) * 5)
    bins = np.arange(min_speed, max_speed + 5, 5)
    
    plt.hist(wind_speeds, bins=bins, edgecolor='black', color='skyblue', align='mid')
    plt.title(f'Distribution of Maximum Wind Speeds for {dataset_name.capitalize()} Hurricanes')
    plt.xlabel('Maximum Wind Speed (knots)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{dataset_name}_wind_speed_distribution.png')
    plt.close()

def plot_hurricane_frequency(df, dataset_name="atlantic"):
    plt.figure(figsize=(12, 6))
    hurricanes = df[df['Status'] == 'HU'][['ID', 'Year']].drop_duplicates()
    yearly_counts = hurricanes.groupby('Year').size()
    
    years = yearly_counts.index
    counts = yearly_counts.values
    
    plt.plot(years, counts, marker='o', color='coral', label='Hurricane Frequency', linestyle='-')
    coefficients = np.polyfit(years, counts, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(years, trendline(years), color='blue', linestyle='--', label=f'Trendline (slope={coefficients[0]:.4f})')
    
    plt.title(f'Unique {dataset_name.capitalize()} Hurricanes per Year with Trendline')
    plt.xlabel('Year')
    plt.ylabel('Number of Hurricanes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{dataset_name}_hurricane_frequency.png')
    plt.close()

def main():
    datasets = {
        'atlantic': 'atlantic.csv',
        'pacific': 'pacific.csv'
    }
    
    for dataset_name, file_path in datasets.items():
        print(f"\nProcessing {dataset_name.capitalize()} Dataset...")
        try:
            data = load_and_clean_data(file_path)
            data['Year'] = data['Datetime'].dt.year
            
            # Check for missing values
            print(f"{dataset_name.capitalize()} Dataset: Records with missing Maximum Wind: {data['Maximum Wind'].isna().sum()}")
            print(f"{dataset_name.capitalize()} Dataset: Records with missing Minimum Pressure: {data['Minimum Pressure'].isna().sum()}")
            wind_radii_cols = ['Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW',
                               'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW',
                               'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
            for col in wind_radii_cols:
                print(f"{dataset_name.capitalize()} Dataset: Records with missing {col}: {data[col].isna().sum()}")
            
            # Generate plots
            plot_hurricane_tracks_on_map(data, min_wind=100, dataset_name=dataset_name)
            plot_wind_speed_distribution(data, dataset_name=dataset_name)
            plot_hurricane_frequency(data, dataset_name=dataset_name)
            
            # Summary statistics
            hurricanes = data[data['Status'] == 'HU']
            avg_wind = hurricanes['Maximum Wind'].mean()
            print(f"{dataset_name.capitalize()} Dataset: Total Hurricanes: {len(hurricanes['ID'].unique())}")
            print(f"{dataset_name.capitalize()} Dataset: Average Maximum Wind Speed: {avg_wind:.2f} knots")
            print(f"{dataset_name.capitalize()} Dataset: Years Covered: {data['Year'].min()} to {data['Year'].max()}")
            print(f"{dataset_name.capitalize()} Dataset: Map plot saved as '{dataset_name}_hurricane_tracks_map.png'")
            print(f"{dataset_name.capitalize()} Dataset: Wind speed distribution saved as '{dataset_name}_wind_speed_distribution.png'")
            print(f"{dataset_name.capitalize()} Dataset: Hurricane frequency plot saved as '{dataset_name}_hurricane_frequency.png'")
        
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Skipping {dataset_name.capitalize()} dataset.")
        except Exception as e:
            print(f"Error processing {dataset_name.capitalize()} dataset: {str(e)}")

if __name__ == "__main__":
    main()
