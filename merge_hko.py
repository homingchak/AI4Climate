import pandas as pd
import os

def merge_hko(directory_path, output_file='HKO_BST.csv'):
    dfs = []

    columns = ['Tropical Cyclone Name', 'Year', 'Month', 'Day', 'Time (UTC)', 'Intensity',
               'Latitude (0.01 degree N)', 'Longitude (0.01 degree E)',
               'Estimated minimum central pressure (hPa)', 'Estimated maximum surface winds (knot)',
               'JMA Code', 'HKO Code'
    ]
    
    for year in range(1985, 2024):
        file_path = os.path.join(directory_path, f'HKO{year}BST.csv')

        if not os.path.exists(file_path):
            print(f"File for year {year} not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path, skiprows=3, header=0, names=columns)   
        dfs.append(df)
        print(f"Loaded data for year {year}: {len(df)} records")
        
    if not dfs:
        print("No data loaded. Exiting.")
        return None
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records after merging: {len(merged_df)}")
    
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    return merged_df

directory_path = 'tropical_cyclone_best_track_data_post_analysis'
merged_df = merge_hko(directory_path)