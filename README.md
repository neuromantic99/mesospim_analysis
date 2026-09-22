# mesospim-analysis

Autofluorescence-corrected cell detection for mesoSPIM light-sheet projections.

Cleared brains carry dense, nucleus-sized autofluorescent puncta (mostly lipofuscin) that are
indistinguishable from labelled cells by size or brightness. They excite broadly, so they also
appear in an off-target channel where the label does not. Imaging the same field at that
second wavelength (e.g. AF647 imaged at 638 nm, autofluorescence at 561 nm) lets them be
subtracted out.

## Data layout

Acquisitions are fused BigStitcher output (BigDataViewer HDF5 + XML), laid out as
`yyyy-mm-dd/MOUSE_ID/IMAGING-NUM/stitched.h5` with `stitched.xml` alongside. Channels are
found through the XML's ViewSetup attributes, and voxel size and pyramid downsampling are read
from the files, so processing parameters (defined in micrometres) follow the pyramid level.

## Usage

Edit the settings block at the top of `main.py` (acquisitions, slab thickness, channels,
depth range), then:

```bash
uv run main.py
```

By default this corrects every consecutive 50 um slab of each listed brain. The functions it
calls live in `mesospim_analysis/run.py` and can also be used from a notebook:

- `run_brain(path, ...)`: whole brain -> BigTIFF stack + per-slab csv; returns the summaries.
- `run_slab(path, z_start_um, ...)`: one slab -> `_bgsub`/`_afcorr` TIFF pair.
- `run_tiff(signal_path, af_path, ...)`: an existing pair of projection TIFFs.
- `acquisitions.find_acquisitions(root, mouse_id=None)`: every stitched.h5 under a root.
- `filters`: separates labelled cells from nonspecific antibody by object geometry.
  `DEFAULT_FILTER.score(rows)` scores objects 0-1, `brain_score(rows)` flags whole brains
  (real brains ~0.67-0.70, a secondary-only control ~0.33), `write_scored_objects` filters an
  existing objects csv, and `fit_nonspecific_filter(real, control)` refits for a new batch.
  Runs on a saved objects csv, so no brain needs reprocessing.
- `plotting.plot_depth_profiles({label: summaries}, path)`: cells, autofluorescence load and
  alpha against depth. `main.py` writes it for every brain it processes (`PLOT_PATH`), and
  `pipeline.read_summary_csv` reads old runs back so figures can be redrawn without reprocessing.

Outputs go to `afcorr/` next to each h5 unless `OUT_DIR` / `out_dir` is set.

- `*_afcorr.tif`: background subtracted with autofluorescence removed. **Count on this.**
- `*_bgsub.tif`: background subtracted only; **still contains autofluorescent puncta**. Kept
  as the "before" image (`SAVE_BGSUB = True`).
- `*_objects.csv` (whole-brain runs, `SAVE_OBJECTS`): one row per detected object, with its
  position, area, peak in each channel, ratio, and where it sits — distance to the nearest dark
  structure (vessel, ventricle, crack) and the length of the structure containing it. Nonspecific
  antibody hugs those boundaries and sits in long thin structures; labelled nuclei are compact and
  at chance distance from them. Adds ~25% to the runtime.
- `*_summary.csv` (whole-brain runs): per slab, the fitted alpha, puncta split into autofluorescent and
  specific, and object counts before/after correction. Slabs that cannot be corrected (e.g.
  no tissue above/below the brain) get a status message and a blank frame, so frame index ==
  slab index.

Count at pyramid level 0: at coarser levels a nucleus is under ~2 px and noise passes as
cells (the tool warns). A nucleus straddling a slab boundary appears in both projections, so
summed slab counts overestimate by up to ~20% (nucleus / slab thickness).

In napari: `load_bigstitched_data` in `utils.py` returns a lazy dask array of one channel.

## Method

1. Local background from a median filter (~98 um window, computed at half resolution),
   subtracted from both channels.
2. The AF channel is soft-thresholded at 3 sigma, so only genuine AF objects are subtracted
   and its shot noise is not injected into the result.
3. `alpha` (signal/AF ratio of autofluorescent objects) is fitted as a robust median over
   pixels with strong autofluorescence.
4. `corrected = bgsub_signal - alpha * softthresh(bgsub_af)`, clipped at zero.

Alpha is fitted **per slab**. It absorbs relative laser power and each brain's AF spectrum
(0.91 vs 0.55 between the first two brains), and 561 and 638 nm light attenuate differently
with depth. Check its stability down the brain in the summary csv.

Puncta (objects >2x local background in the signal channel) are classified by their
background-subtracted signal/AF peak ratio: autofluorescence sits near 0.7, while AF647 bleeds
into the 561 nm channel at ~7%, putting labelled cells near 14. The cutoff is 2. Classifying on
whether an object is visible in the AF channel at all discards the brightest labelled cells,
because bleed-through makes them visible there.

In the summary csv, `tissue_px` is ~60% of the frame in every slab and does not indicate
tissue; `n_fit_pixels` (autofluorescent pixels found) does. Slabs with a few hundred fit pixels
or fewer are effectively outside the brain and their alpha is unreliable.

BigDataViewer stores uint16 as int16; this is undone on read so bright pixels are not lost
from max projections.

Both channels must come from the same acquisition so they are pixel-registered.

## Nonspecific antibody

The autofluorescence correction cannot remove secondary antibody bound where it should not be:
that is genuine AF647. A secondary-only control measures it directly — in N041 it was ~18% of a
real brain's object density. Those objects accumulate on vessel, ventricle and tissue surfaces
and sit in long thin structures, while labelled nuclei are compact and at chance distance from
such boundaries. `filters` scores that difference (AUC 0.83 measured, ~0.90 once the real
brain's own nonspecific fraction is accounted for). Keeping 95% of a real brain's objects
removes 44% of a control's, taking the nonspecific fraction from ~18% to ~10%.

Object brightness is deliberately excluded: it separates the populations within one brain but
reflects that brain's exposure, and including it made the model worse on a held-out brain.
Nonspecific binding depends on secondary lot and blocking, so refit per staining batch.

## Validation

On a secondary-only control brain, correction removes 98.8% of objects above 200 ADU
(849 -> 10), which sets the false-positive floor. On a stained brain it keeps 98.4% (506 -> 498).

## Development

```bash
uv run mypy      # strict
uv run pytest
```
