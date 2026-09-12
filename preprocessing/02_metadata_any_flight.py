#!/usr/bin/env python3
"""
02_metadata_any_flight.py
=========================
Step 11, parameterised. Does exactly what 02_metadata.py does, for either
flight, instead of only the one its path happens to point at.

    python 02_metadata_any_flight.py 1 /path/to/data_processed
    python 02_metadata_any_flight.py 2 /path/to/data_processed

WHICH SCRIPT WAS ACTUALLY RUN: 02_metadata.py, the hardcoded one, twice --
once as written for Flight 1, then again after editing the path to Flight 2.
This file did not exist at the time and produced none of the thesis data. It
is here so the same code covers both flights without editing a path, and so
the Flight 2 run is represented by something other than an edit nobody wrote
down. See README.md, step 11.

Like everything in this folder, it needs the cluster and the raw imagery, so
it cannot be run from this repository.
"""
import sys
from pathlib import Path

BAND_NAMES = ["Green", "Red", "RedEdge", "NIR"]   # the stack's band order, fixed at step 9


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <flight 1|2> <data_processed directory>")
    flight, data_dir = sys.argv[1], Path(sys.argv[2])
    if flight not in {"1", "2"}:
        sys.exit(f"flight must be 1 or 2, not {flight!r}")

    raster = data_dir / f"multispectral_{flight}.tif"
    if not raster.exists():
        sys.exit(f"not found: {raster}")

    from osgeo import gdal      # imported here so --help style misuse fails without GDAL

    ds = gdal.Open(str(raster), gdal.GA_Update)
    if ds is None:
        sys.exit(f"GDAL could not open {raster}")
    if ds.RasterCount != len(BAND_NAMES):
        sys.exit(f"{raster} has {ds.RasterCount} bands, expected {len(BAND_NAMES)}")

    for i, name in enumerate(BAND_NAMES, start=1):
        band = ds.GetRasterBand(i)
        band.SetDescription(name)
        # True = approximate statistics, as in the original. These are display
        # metadata only; nothing downstream reads them. See README.md, step 11.
        band.ComputeStatistics(True)

    ds = None
    print(f"Metadata added to {raster.name}: " + ", ".join(BAND_NAMES))


if __name__ == "__main__":
    main()
