import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
import numpy as np
import os

tif_path = r'./data/xgboost_stil_predict_output/grid/tif'
tif_files = os.listdir(tif_path)
index = pd.read_csv('./data/grid_index.csv')

for i, tif_file in enumerate(tif_files):

    # Read the TIF file
    with rasterio.open(os.path.join(tif_path, tif_file)) as src:
        raster = src.read(1)
        transform = src.transform

    # Get the height and width of the image
    height, width = raster.shape

    # Calculate row and column indices for each grid cell
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Calculate the geographic coordinates of the centroid of each pixel
    x_coords, y_coords = transform * (cols + 0.5, rows + 0.5)

    # Calculate the grid index for each cell
    indices = np.ravel_multi_index((rows, cols), (height, width))

    # Save the results as a DataFrame
    df = pd.DataFrame({
        'centroid_x': x_coords.ravel(),
        'centroid_y': y_coords.ravel(),
        'grid_id': indices.ravel(),
    })

    # Merge the electricity data with the grid data
    filtered_results_df = pd.merge(index[['grid_id', 'county']], df, on='grid_id', how='left')
    filtered_results_df.rename(columns={'county': 'district'}, inplace=True)

    # Calculate the number of grids in each district
    district_counts = filtered_results_df['district'].value_counts()

    # Calculate weights and add them to the DataFrame
    filtered_results_df['weight'] = 1 / filtered_results_df['district'].map(district_counts)

    # Save the results to a CSV file
    output_csv_path = os.path.join(r'./krige_output', f"{tif_file.split('.')[0]}_point_weight_centroid.csv")
    filtered_results_df.to_csv(output_csv_path, encoding='utf-8', index=False)
