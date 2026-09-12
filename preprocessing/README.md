# preprocessing/ — the steps that ran outside this repository

> **Nothing in this folder can be run from this repository.**
> These steps ran on the University of Birmingham BlueBEAR HPC cluster and inside the QGIS
> desktop application, against raw UAV imagery that is not distributed here and is far too large
> to be. They need the cluster, its module system, and the original orthomosaics.
> **They are included for documentation only** — so that the provenance of
> `ALL_stats_1_JB_jack_data.csv` and `ALL_stats_2_JB_jack_data.csv`, which are where the Python
> pipeline begins, is written down rather than remembered.
>
> They are deliberately **not** wired into `reproduce.py`. `reproduce.py` only runs things that
> can actually be rerun and checked here, and none of this can be.

## Environment

Loaded on BlueBEAR before anything else:

```
module load bear-apps/2023a/live
module load QGIS/3.40.2-foss-2023a
```

That is the whole environment for these steps: the `bear-apps/2023a/live` module bundle, and
QGIS 3.40.2 built against `foss-2023a`, which brings the GDAL used by both the command-line
tools and the QGIS processing algorithms.

## Coordinate reference systems

Three CRSs appear in the QGIS project, and they are not interchangeable:

| CRS | Name | Where it is used |
|---|---|---|
| `EPSG:4326` | WGS 84 | the QGIS **project CRS**, and the CRS of most layers |
| `EPSG:27700` | OSGB36 / British National Grid | the plot grid and the layers aligned to it |
| `EPSG:3857` | WGS 84 / Pseudo-Mercator | the web basemap only |

`EPSG:4326` is a geographic CRS in degrees, `EPSG:27700` a projected one in metres. Zonal
statistics are sampled inside `FINAL_grid.gpkg` plot boundaries, so the grid's CRS is the one
that governs which pixels belong to which plot. A CRS transformation question was open in the
lab notebook at the point these steps finished; it is listed there as a question for the
supervisor, not as a resolved decision.

## Files here

| File | What it is |
|---|---|
| `01_preprocessing.sh` | steps 2 and 3, as run. **Redacted** — see below |
| `02_metadata.py` | step 11, as written. **Redacted** — see below |
| `02_metadata_any_flight.py` | step 11, rewritten to take the flight number as an argument. **Not the script that was run** — see step 11 |
| `potato_pipeline.qgz` | the QGIS project carrying steps 1 and 6 to 15. **Not in this repository — available on request**, see below |
| `16_texture_extraction_bluebear.ipynb` | step 16, the texture extraction run on BlueBEAR. **Outputs cleared, paths redacted** |

### Four of these files are REDACTED and are therefore not what ran

**`01_preprocessing.sh`, `02_metadata.py`, `potato_pipeline.qgz` and
`16_texture_extraction_bluebear.ipynb` as committed here are NOT byte-identical to the files that
produced the data.** Each carried identifying strings that cannot be published, and those strings
have been replaced with clearly marked placeholders. Nothing else was touched: no logic, no
parameter, no threshold, no ordering. The unredacted originals are kept outside this repository
and are the authoritative record of what ran.

| Placeholder | What it stands for |
|---|---|
| `__REDACTED_CLUSTER_PROJECT__` | the absolute path to the shared cluster project directory |
| `__REDACTED_WORKSPACE__` | the author's working directory inside it |
| `__REDACTED_CLUSTER_HOME__` | the author's cluster home directory |
| `__REDACTED_USERNAME__` | the author's university username |
| `__REDACTED_SUPERVISOR__` | a supervisor's username, which appears inside the project directory name |
| `__REDACTED_NAME__` | the author's name where it appeared in a path |

What this means in practice:

- **None of these scripts will run as committed.** They could not run here anyway, for the
  reasons at the top of this file, but the placeholders make that unambiguous rather than
  leaving someone to discover it when a path fails.
- **The QGIS project will open but its layers will not resolve**, because the redaction also
  rewrote the layer identifiers QGIS derives from file paths. It is committed to document the
  processing chain, not to be opened and used.
- **Line counts, ordering and every parameter are unchanged**, so the files remain accurate as a
  record of method even though they are inaccurate as a record of bytes.

`README.md` and `02_metadata_any_flight.py` contained none of these strings and are committed
unmodified.

### The debug cells in the notebook are kept deliberately

The leak scanner raises two REVIEW items against
`16_texture_extraction_bluebear.ipynb`, both on debug cells that filter to a single plot by
its ID while checking that the freshly extracted pixel statistics line up with the existing zonal
statistics for the same plot. A single plot ID in a debug cell is not a data leak, and the cells are part of how the
extraction was verified. They stay as they were run. REVIEW items are for reading, not for
automatic fixing, and this is one that was read and accepted.

### `potato_pipeline.qgz` is redacted but still NOT committed

Redacting the paths dealt with the usernames. It did not deal with what else a QGIS project
records. The file stores the map extent of the trial:

```
<extent>
  <xmin>-1.908...</xmin>  <ymin>54.987...</ymin>
  <xmax>-1.906...</xmax>  <ymax>54.988...</ymax>
</extent>
```

That is a box roughly 200 m by 130 m, in WGS 84, which locates the field site to within a few
metres. It is in the project whether or not any layer resolves, and it cannot be redacted the way
a file path can without making the project meaningless.

The file carries **no trial measurements** — every one of its 50 layers is a `gdal` reference to
an external raster and there are no embedded or memory layers, so no disease score, plot value or
variety name is inside it. The exposure is the location of the trial, not its contents.

Publishing the site's coordinates is a disclosure about the trial rather than about the author,
and it is not something the repository needs: **code availability does not require the project
file.** Everything the project did is described in the step table above — the layers it builds,
the thresholds it applies, the order it applies them in — and that description is what makes the
method reproducible for anyone who has the imagery.

**The QGIS project is therefore not included in this repository. It is available on request**,
alongside the imagery itself, under the same access conditions as the rest of the trial data. See
[Data availability](../README.md#data-availability) in the top-level README.

It is listed explicitly in `.gitignore`, not merely covered by the blanket `*.qgz` rule, so that
relaxing that rule later cannot quietly pull it in.


## The numbered steps, from the lab notebook

| Step | What was done | Status |
|---|---|---|
| 1 | Load `RGB_1.tif` as a raster layer | done |
| 2 | Remove alpha bands: keep band 1 only, for Green, Red, NIR and RedEdge, both flights | done |
| 3 | Rescale to reflectance: divide by 32768, the Metashape 16-bit integer convention | done |
| 4 | Plot grid — `FINAL_grid.gpkg` already provided | pre-existing, no action |
| 5 | Re-labelling the grid — `FINAL_grid.gpkg` already carries `Variety_ID` | pre-existing, no action |
| 6 | Load the plot grid into QGIS | done in the project |
| 7A | ExG from RGB, normalised: `2*(G/(R+G+B)) - (R/(R+G+B)) - (B/(R+G+B))` | done |
| 7 | Soil mask: threshold ExG at **0.2**, then sieve at **threshold 20, 8-connectedness**, no validity mask | done |
| 8 | Apply the sieved soil mask to each rescaled band: `"band_rescaled@1" / "soil_mask_sieved@1"`, so soil (0) becomes NoData and vegetation (1) keeps its reflectance | done |
| 9 | Build a virtual raster stacking the four masked bands, one file per band | done |
| 10 | Translate the `.vrt` to a permanent GeoTIFF | done |
| 11 | Add band names and compute band statistics (`02_metadata.py`) | done |
| 12 | Clip to the field area | **skipped, justified — see below** |
| 13–15 | Vegetation indices, canopy cover, zonal statistics per plot, merge and export to `ALL_stats_<flight>.csv` | done |

Steps 7 to 15 were carried out for Flight 1 first and then repeated for Flight 2.

Step 16, texture extraction, ran later and separately, in a notebook on BlueBEAR rather than in
QGIS. It is documented in its own section below.


### Thresholds, fixed and not tuned

- **ExG threshold 0.2**, for both flights. Recorded in the lab notebook as decided by Jack
  Bosanquet, not selected by looking at model performance.
- **Sieve threshold 20, 8-connectedness, no validity mask.** Removes clumps smaller than 20
  connected pixels from the soil mask, counting diagonal neighbours as connected.

### Band order

**Green, Red, RedEdge, NIR** — set when the virtual raster was built at step 9 and named at
step 11. Anything reading these stacks by band index depends on this order.

### Step 12 was skipped, and why

Clipping to the field area was skipped entirely, for both flights. The lab notebook records the
reasoning:

- The field technician's slide 19 lists the step as optional.
- No field boundary shapefile existed in the shared data folder.
- An attempt to draw one by hand ended when QGIS ran out of memory and crashed.
- **Justification given:** the zonal statistics at step 13 only sample pixels inside
  `FINAL_grid.gpkg` plot boundaries, so pixels outside the field are excluded whether or not the
  raster was clipped first.
- **Impact recorded:** none.

That justification is sound for the statistics actually computed, since every number that
reaches the Python pipeline is a per-plot zonal statistic. It is worth stating explicitly
because "skipped" and "skipped, and here is why it does not matter" are different claims, and
this is the second.

## Two discrepancies recorded rather than smoothed over

### 1. Step 3 exists in two versions that do not compute the same thing

There is a shell version and a Python version of steps 2 and 3. They agree on step 2 and
**disagree on step 3**:

| | Step 3 as written | Behaviour above 32768 |
|---|---|---|
| **Shell** (`01_preprocessing.sh`) | `gdal_calc.py --calc="A/32768.0"` | divides, **does not clamp**; a DN of 40000 becomes 1.22 |
| **Python** (`01_preprocessing.py`, not in this folder) | `gdal.Translate(..., scaleParams=[[0, 32768, 0, 1]])` | linear rescale of the range 0–32768 onto 0–1, and GDAL **clamps** to the destination range, so every DN above 32768 becomes exactly 1.0 |

Below 32768 the two agree. Above it they do not, and reflectance above 1.0 is physically
possible in this data (specular glint, bright targets), so the clamp is not a harmless tidy-up:
it would flatten every over-range pixel to the same value and erase the differences between them.

**The lab notebook records the shell version as the one used** — step 3 says "Used `gdal_calc.py`
from terminal". So the unclamped division is what produced the reflectance rasters behind the
thesis. The Python variant is kept out of this folder to avoid any suggestion that either could
be run interchangeably. If any figure or number is ever traced back to the Python variant, it is
from a different pipeline than the one recorded.

### 2. This QGIS project computes five vegetation indices; the modelling data has eleven

`potato_pipeline.qgz` computes and takes zonal statistics for **five** indices:

> NDVI, NDRE, GNDVI, EVI2, OSAVI

The tables the Python pipeline actually reads, `ALL_stats_<flight>_JB_jack_data.csv`, carry
**eleven**: the five above plus RDVI, PVR, TVI, TCARI, TCARI_OSAVI and canopy cover.

So this project file is **not** the thing that produced the modelling inputs. It documents the
route through QGIS — masking, stacking, zonal statistics — but the eleven-index tables came from
a separate pipeline, reflected in those files' `JB_jack_data` naming. Reading this project and
assuming it generated the modelling features would be wrong. Which pipeline produced the
eleven-index tables, and whether it repeats the same masking and thresholds, is not documented
anywhere I can see and is worth pinning down: every feature in the locked model comes from those
tables, and their provenance currently stops at a filename.

## Step 11 in detail

`02_metadata.py` sets the four band descriptions and computes band statistics.

**It is hardcoded to Flight 1.** The raster path ends `multispectral_1.tif`. Flight 2 therefore
required editing that path to `multispectral_2.tif` and rerunning the script by hand. **That edit
is not recorded anywhere** — not in the lab notebook, which lists step 11 once and marks it done,
and not in the script, which survives only in its Flight 1 form. The Flight 2 run is inferred
from the fact that the Flight 2 stack carries band names, not from any record of it happening.
`02_metadata_any_flight.py` is here so that the same code covers both flights without an
unrecorded edit; **it is not the script that was run**, and it produced none of the thesis data.

**`ComputeStatistics(True)` computes approximate statistics, not exact ones.** The `True` is
GDAL's `approx_ok` flag: it permits statistics estimated from overviews or a subsample rather
than a full pass over every pixel. Minimum, maximum, mean and standard deviation may therefore
differ slightly from the true values. This does not affect any result in the thesis: these
statistics are **display metadata**, used by QGIS to pick default contrast stretches, and nothing
downstream reads them. Every number in the analysis comes from the zonal statistics at step 13,
which are computed from pixel values, not from this metadata.

## Step 16 — texture extraction on BlueBEAR

`16_texture_extraction_bluebear.ipynb`, run on the cluster with **Python 3.12.3**, using
**rasterio** to read the masked band rasters and **geopandas** to read the plot grid. Its outputs
are cleared in the copy here; on the cluster it ran with outputs.

What it does:

1. Reads `FINAL_grid.gpkg` and **reprojects the plot polygons from EPSG:27700 to EPSG:4326**, so
   they match the CRS the masked rasters are written in. Without that step the polygons and the
   pixels do not line up at all.
2. For each of the four bands (Green, Red, RedEdge, NIR) and each of the two flights, cuts each
   plot polygon out of the masked raster and computes **five percentile statistics per band per
   flight**: `p10`, `p90`, `std`, `min`, `max`. Soil pixels are already NoData from step 8, so
   the statistics describe vegetation only. That is 4 bands x 2 flights x 5 statistics = 40
   feature columns.
3. **Drops the guard plots** by requiring `Disease_Plot_ID` to be entirely numeric
   (`^\d+$`); the grid's non-numeric IDs are border guards, not scored plots.
4. Writes **`texture_features_percentiles.csv`** (all 40 columns), then a three-feature subset,
   **`texture_features_selected.csv`**.

Both files are inputs to the Python pipeline: `texture_features_selected.csv` supplies the three
texture features in the locked model, and `texture_features_percentiles.csv` supplies the twenty
per-flight texture columns used by the variety-level and Kruskal-Wallis analyses.

### Features that were computed and then dropped

- **Coefficient of variation** (`<band>_<flight>_cv_range`, computed as
  `(p90 - p10) / ((p90 + p10) / 2)`) was computed for all eight band-flight combinations and then
  **dropped, because its direction flipped between flights** — a plot type that looked patchier
  in one flight looked smoother in the other, so the feature was not measuring a stable property.
- **GLCM features** (contrast, homogeneity, energy, correlation, 16 grey levels, four directions
  averaged, with a NaN-aware implementation written specially because `skimage.graycomatrix`
  cannot handle masked pixels) were computed and then **dropped** as well.

Neither set reaches `texture_features_percentiles.csv`. Both are worth knowing about: a reader
who sees only the surviving five statistics would not guess how much was tried and discarded.

### The three selected texture features were chosen using labels from both flights

This is the important one, and it is a leakage problem, not a detail.

The selection ranks all twenty band-and-statistic combinations by how well they separate diseased
from healthy plots. For each candidate it computes the diseased-minus-healthy difference in
**Flight 1** and in **Flight 2**, checks whether the two differences point the **same direction**,
and ranks by the average of the two normalised effect sizes.

**Flight 1 is the held-out external test flight.** Its labels are supposed to be unseen until the
final evaluation. They were used here, to choose which features the locked model would be built
from. So the three texture features were selected on the full dataset, test flight included, and
the external result in `02_locked_model.py` is not as clean a held-out number as it looks. The
effect is likely small — three features out of seven, chosen by a coarse criterion — but it is
real, and it is in the direction that flatters the result. `src/07_selection_leakage_check.py`
measures it directly by redoing the selection with Flight 2 labels alone.

### The stated criterion does not by itself pick the three that were used

The consistency criterion is passed by **fourteen of the twenty** candidates, so it narrows the
field but decides nothing on its own. Ranked by average normalised effect, the top three
consistent candidates are `NIR_max`, `NIR_p90` and `RedEdge_max`. The three actually carried
forward were **`NIR_max`, `RedEdge_p90` and `Red_min`**.

Reconstructing the rule that produces that set, two further constraints were applied but never
written down:

- **one statistic per band, and Green excluded** — leaving Red, RedEdge and NIR, which rules out
  `NIR_p90` (NIR already used by `NIR_max`);
- **each statistic used at most once** — which rules out `RedEdge_max`, since `max` was already
  taken by NIR, and takes `RedEdge_p90` instead, even though `RedEdge_max` ranks above it
  (0.387 against 0.368).

`Red_min` is the top-ranked Red candidate, so it needs no special explanation. The point is that
the written criterion is not sufficient to reproduce the selection: two unstated rules are doing
part of the work. Anyone re-deriving these three features from the stated method alone will get a
different set.

## Provenance, end to end

```
raw orthomosaics (cluster, not distributed)
   -> step 2,3   01_preprocessing.sh          alpha removed, divided by 32768
   -> step 7     ExG > 0.2, sieved (20, 8-connected)      soil mask
   -> step 8     mask applied to each band
   -> step 9,10  4-band stack, Green Red RedEdge NIR
   -> step 11    02_metadata.py               band names + approximate stats
   -> step 13-15 zonal statistics per plot
   -> ALL_stats_<flight>_JB_jack_data.csv     <- the Python pipeline starts here

   -> step 16    16_texture_extraction_bluebear.ipynb   (rasterio + geopandas, Python 3.12.3)
                 plot polygons reprojected EPSG:27700 -> EPSG:4326
                 5 percentile stats per band per flight, guard plots dropped
   -> texture_features_percentiles.csv        <- also an input to the Python pipeline
   -> texture_features_selected.csv           <- the three features in the locked model
```

Everything above the last line happened on the cluster and in QGIS and cannot be rerun here.
Everything from that line down is in `src/` and is checked by `reproduce.py`.
