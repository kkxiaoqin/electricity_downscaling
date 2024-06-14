import numpy as np
from osgeo import gdal
import pandas as pd
import os

def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    Convert an array to a raster file and save it to disk.

    :param arr: Input array to be converted to raster.
    :param raster_file: Path to the output raster file.
    :param prj: Projection information, obtained via gdal's GetProjection(), default is None.
    :param trans: Geotransformation information, obtained via gdal's GetGeoTransform(), default is None.
    """
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float64)

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    # Write the array to the raster band
    dst_ds.GetRasterBand(1).WriteArray(arr)

    dst_ds.FlushCache()
    dst_ds = None

if __name__ == '__main__':
    # Path to the directory containing CSV files with point data
    point_csv_path = r'./data/xgboost_stil_predict_output/grid'
    point_csv_files = os.listdir(point_csv_path)

    # Iterate through each CSV file in the directory
    for point_csv_file in point_csv_files:
        # Read the CSV file
        ele = pd.read_csv(os.path.join(point_csv_path, point_csv_file), encoding='utf_8_sig', index_col=0)
        ele = ele.iloc[:, -1]
        index = ele.index
        grid = np.zeros((5255, 4833)) # Determined by the number of Chinese rasters covered

        # Fill the grid array with values from the CSV file
        for j in index:
            row = int(j) // 4833
            col = int(j) % 4833
            grid[row][col] = ele.loc[j]

        # Define the output array and raster file path
        arr = grid
        raster_file = os.path.join(r'./data/xgboost_stil_predict_output/grid',
                                   f"{point_csv_file.split('.')[0]}.tif") # Here is just one example, the result is named "predict_csv2tif_demo.tif" in code.


        # Path to the source raster file for geographical and geometrical information
        src_ras_file = r'./data/county.shp'
        dataset = gdal.Open(src_ras_file)
        projection = dataset.GetProjection()
        transform = dataset.GetGeoTransform()

        # Convert the array to a raster file
        arr2raster(arr, raster_file, prj=projection, trans=transform)
