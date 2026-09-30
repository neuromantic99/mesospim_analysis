"""Paths that differ between machines.

The mesoSPIM data is on a different mount on the server than on the laptop, so both roots are
read from the environment. The defaults are the laptop's, which is where analysis is usually
run; on the server, export MESOSPIM_DATA_ROOT and MESOSPIM_RESULTS_DIR instead of editing this
file, so the import does not break for whoever has the other mount.
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/Volumes/MarcBusche/James/Mesospim")
"""Acquisitions, as yyyy-mm-dd/MOUSE_ID/IMAGING-NUM/stitched.h5."""

DEFAULT_RESULTS_DIR = Path("/Volumes/hard_drive/Mesospim/afcorr_results")
"""Where copy_results gathers each acquisition's summary and objects CSVs."""

DATA_ROOT = Path(os.environ.get("MESOSPIM_DATA_ROOT") or DEFAULT_DATA_ROOT)
RESULTS_DIR = Path(os.environ.get("MESOSPIM_RESULTS_DIR") or DEFAULT_RESULTS_DIR)

AFCORR_DIRNAME = os.environ.get("MESOSPIM_AFCORR_DIRNAME") or "afcorr_c15"
"""Subfolder of each acquisition holding its results, and what copy_results gathers.

Named for the detection contrast it was produced at. The first cohort ran at 2.0 into
`afcorr/`; re-running into the same folder would overwrite it, and the two are not comparable
because lowering the threshold roughly triples the counts."""
