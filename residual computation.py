import numpy as np
import pandas as pd
import geopandas as gpd
import os

# Calculate residuals between county and grid data for subsequent Kriging interpolation

# Step 1: Read county shapefile and initialize paths for county and grid prediction files
gdf = gpd.read_file(r'./data/county.shp') # Vector data of Chinese county boundaries.

# Paths to county and grid prediction folders
county_path = r'data/xgboost_stil_predict_input/county'
county_files = os.listdir(county_path)

grid_path = r'data/xgboost_stil_predict_input/grid'
grid_files = os.listdir(grid_path)


# Iterate through each grid file to calculate residuals and save to shapefile
for i, grid_file in enumerate(grid_files):
    # Read county and grid CSV files
    county_csv = pd.read_csv(os.path.join(county_path, county_files[i]), encoding='utf_8_sig', index_col=0)
    grid_csv = pd.read_csv(os.path.join(grid_path, grid_file), encoding='utf_8_sig')

    # Group grid data by county and sum the values
    grid_csv_count = grid_csv.groupby('county').sum()

    # Merge county and grid data on the county index
    df_merge = pd.merge(county_csv, grid_csv_count, how='inner', left_index=True, right_index=True)
    df_merge = df_merge[['month_county_weight_predict', 'predict_y']]

    # Calculate residuals
    df_merge['residual'] = df_merge['month_county_weight_predict'] - df_merge['predict_y']
    df_merge['index'] = df_merge.index
    df_merge = df_merge.reset_index(drop=True)

    # Rename columns for clarity
    df_merge = df_merge.rename(columns={'index': 'PAC', 'predict_y': 'predict', 'residual': grid_file.split('.')[0]})
    df_merge = df_merge.drop(['month_county_weight_predict', 'predict'], axis=1)

    # Merge residuals with the original shapefile data
    if i == 0:
        merged = gdf.merge(df_merge, on='PAC', how='left')
    else:
        merged = merged.merge(df_merge, on='PAC', how='left')

    # Fill any NaN values with 0
    merged.fillna(0, inplace=True)

# Save the merged shapefile with residuals
merged.to_file(r'./data/residual_shp.shp', encoding='utf-8')
