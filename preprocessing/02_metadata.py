from osgeo import gdal

raster = "__REDACTED_CLUSTER_PROJECT__/__REDACTED_WORKSPACE__/data_processed/multispectral_1.tif"

band_names = ["Green", "Red", "RedEdge", "NIR"]

ds = gdal.Open(raster, gdal.GA_Update)

for i, name in enumerate(band_names, start=1):
    band = ds.GetRasterBand(i)
    band.SetDescription(name)
    band.ComputeStatistics(True)

ds = None
print("Metadata added successfully.")