import pandas as pd
import numpy as np
from osgeo import gdal

# Read data
residual = pd.read_csv(r'./data/krige_output/point_weight_centroid_demo.csv', encoding='utf_8_sig')
predict = pd.read_csv(r'./data/xgboost_stil_predict_output/grid/csv/output_grid_demo.csv', encoding='utf_8_sig')
predict.rename(columns={'no': "grid_id"}, inplace=True)

# Calculate result and merge data
residual['result'] = residual['pred'] / residual.groupby('areaId')['areaId'].transform('count')
predict_revise = pd.merge(residual, predict, on='grid_id')
predict_revise['revise_data'] = predict_revise['result'] + predict_revise['predict']

# Normalize and calculate weights
group_stats = predict_revise.groupby('areaId')['revise_data'].agg(['max', 'min', 'sum'])
predict_revise = predict_revise.join(group_stats, on='areaId')
predict_revise['normalized_revise'] = (predict_revise['revise_data'] - predict_revise['min']) / (predict_revise['max'] - predict_revise['min'])
predict_revise['weight'] = predict_revise['normalized_revise'] / predict_revise.groupby('areaId')['normalized_revise'].transform('sum')
predict_revise['new_revise'] = (predict_revise['weight'] * predict_revise['sum']).abs().fillna(0)

# Save results to CSV
predict_revise.set_index('grid_id', inplace=True)

# Function to convert array to raster file
def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    Convert array to raster file and save to disk.
    :param arr: Input array
    :param raster_file: Output raster file path
    :param prj: Projection information (optional)
    :param trans: Geotransformation information (optional)
    """
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float64)
    if prj: dst_ds.SetProjection(prj)
    if trans: dst_ds.SetGeoTransform(trans)
    dst_ds.GetRasterBand(1).WriteArray(arr)
    dst_ds.FlushCache()
    dst_ds = None

# Create grid array
grid = np.zeros((5255, 4833))
for j in predict_revise.index:
    row, col = divmod(j, 4833)
    grid[row, col] = predict_revise.loc[j, 'new_revise']

# Get projection and geotransformation information
src_ras_file = r'./data/county.shp'
dataset = gdal.Open(src_ras_file)
projection = dataset.GetProjection()
transform = dataset.GetGeoTransform()

# Save array as TIF
raster_file = r'./data/krige_output/final_result_demo.tif'
arr2raster(grid, raster_file, prj=projection, trans=transform)
