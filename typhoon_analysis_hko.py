import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import load_and_clean_data_hko

def plot_typhoon_tracks_on_map(df, min_wind=100, dataset_name="hko"):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_extent([100, 180, -10, 50], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    
    strong_typhoons = df[(df['Intensity'].isin(['T', 'ST', 'SuperT'])) & 
                        (df['Estimated maximum surface winds (knot)'] >= min_wind)]
    print(f"{dataset_name.upper()} Dataset: Number of records with Intensity in ['T', 'ST', 'SuperT'] and Estimated maximum surface winds >= {min_wind}: {len(strong_typhoons)}")
    print(f"{dataset_name.upper()} Dataset: Number of typhoons with Intensity in ['T', 'ST', 'SuperT'] and Estimated maximum surface winds >= {min_wind}: {len(strong_typhoons['Typhoon_ID'].unique())}")
    
    num_typhoons = len(strong_typhoons['Typhoon_ID'].unique())
    if num_typhoons == 0:
        print(f"Warning: No typhoons meet the criteria for plotting tracks (Intensity in ['T', 'ST', 'SuperT'] and Estimated maximum surface winds >= {min_wind}) in {dataset_name.upper()} dataset.")
        plt.title(f"Tracks of {dataset_name.upper()} Typhoons with Estimated Maximum Surface Winds ≥ {min_wind} knots on Map (No Data)")
        plt.savefig(f'{dataset_name}_typhoon_tracks_map.png')
        plt.close()
        return
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_typhoons, 20)))
    color_cycle = np.repeat(colors, np.ceil(num_typhoons / 20).astype(int), axis=0)[:num_typhoons]
    
    for idx, typhoon_id in enumerate(strong_typhoons['Typhoon_ID'].unique()):
        typhoon_data = strong_typhoons[strong_typhoons['Typhoon_ID'] == typhoon_id]
        typhoon_data = typhoon_data.sort_values('Datetime')
        
        if len(typhoon_data) < 2:
            print(f"{dataset_name.upper()} Dataset: Typhoon {typhoon_id}: Not enough points to plot (need at least 2 points).")
            continue
        
        ax.plot(typhoon_data['Longitude'], typhoon_data['Latitude'],
                color=color_cycle[idx], linewidth=2, alpha=0.7, transform=ccrs.PlateCarree())
    
    plt.title(f'Tracks of {dataset_name.upper()} Typhoons with Estimated Maximum Surface Winds ≥ {min_wind} knots on Map')
    plt.savefig(f'{dataset_name}_typhoon_tracks_map.png')
    plt.close()

def plot_wind_speed_distribution(df, dataset_name="hko"):
    plt.figure(figsize=(10, 6))
    typhoons = df[df['Intensity'].isin(['T', 'ST', 'SuperT'])]
    wind_speeds = typhoons['Estimated maximum surface winds (knot)'].dropna()
    
    min_speed = int(np.floor(wind_speeds.min() / 5) * 5)
    max_speed = int(np.ceil(wind_speeds.max() / 5) * 5)
    bins = np.arange(min_speed, max_speed + 5, 5)
    
    plt.hist(wind_speeds, bins=bins, edgecolor='black', color='skyblue', align='mid')
    plt.title(f'Distribution of Estimated Maximum Surface Winds for {dataset_name.upper()} Typhoons')
    plt.xlabel('Estimated Maximum Surface Winds (knots)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{dataset_name}_wind_speed_distribution.png')
    plt.close()

def plot_typhoon_frequency(df, dataset_name="hko"):
    plt.figure(figsize=(12, 6))
    typhoons = df[df['Intensity'].isin(['T', 'ST', 'SuperT'])][['Typhoon_ID', 'Year']].drop_duplicates()
    yearly_counts = typhoons.groupby('Year').size()
    
    years = yearly_counts.index
    counts = yearly_counts.values
    
    plt.plot(years, counts, marker='o', color='coral', label='Typhoon Frequency', linestyle='-')
    coefficients = np.polyfit(years, counts, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(years, trendline(years), color='blue', linestyle='--', label=f'Trendline (slope={coefficients[0]:.4f})')
    
    plt.title(f'Unique {dataset_name.upper()} Typhoons per Year with Trendline')
    plt.xlabel('Year')
    plt.ylabel('Number of Typhoons')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{dataset_name}_typhoon_frequency.png')
    plt.close()

def main():
    datasets = {
        'hko': 'HKO_BST.csv'
    }
    
    for dataset_name, file_path in datasets.items():
        print(f"\nProcessing {dataset_name.upper()} Dataset...")
        try:
            data = load_and_clean_data_hko(file_path)
            data['Year'] = data['Datetime'].dt.year

            print(f"{dataset_name.upper()} Dataset: Total Records: {len(data)}")
            
            # Generate plots
            plot_typhoon_tracks_on_map(data, min_wind=100, dataset_name=dataset_name)
            plot_wind_speed_distribution(data, dataset_name=dataset_name)
            plot_typhoon_frequency(data, dataset_name=dataset_name)
            
            # Summary statistics
            typhoons = data[data['Intensity'].isin(['T', 'ST', 'SuperT'])]
            avg_wind = typhoons['Estimated maximum surface winds (knot)'].mean()
            print(f"{dataset_name.upper()} Dataset: Total Typhoons: {len(typhoons['Typhoon_ID'].unique())}")
            print(f"{dataset_name.upper()} Dataset: Average Estimated Maximum Surface Winds: {avg_wind:.2f} knots")
            print(f"{dataset_name.upper()} Dataset: Years Covered: {data['Year'].min()} to {data['Year'].max()}")
            print(f"{dataset_name.upper()} Dataset: Map plot saved as '{dataset_name}_typhoon_tracks_map.png'")
            print(f"{dataset_name.upper()} Dataset: Wind speed distribution saved as '{dataset_name}_wind_speed_distribution.png'")
            print(f"{dataset_name.upper()} Dataset: Typhoon frequency plot saved as '{dataset_name}_typhoon_frequency.png'")
        
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Skipping {dataset_name.upper()} dataset.")
        except Exception as e:
            print(f"Error processing {dataset_name.upper()} dataset: {str(e)}")

if __name__ == "__main__":
    main()