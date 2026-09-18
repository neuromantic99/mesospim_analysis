from pathlib import Path

import numpy as np
import tifffile

from mesospim_analysis.types import FloatImage
from mesospim_analysis.utils import as_unsigned


def load_projection(path: Path) -> FloatImage:
    """Load a single-page projection TIFF as float32, undoing BigDataViewer's int16 storage."""
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"{path}: expected a 2D projection, got shape {image.shape}")
    return as_unsigned(image).astype(np.float32)


def load_channel_pair(signal_path: Path, af_path: Path) -> tuple[FloatImage, FloatImage]:
    """Load the signal and autofluorescence channels of one brain, cropped to their common shape.

    The two channels must come from the same acquisition, so they are assumed to be
    pixel-registered; only an off-by-one edge from the export is tolerated.
    """
    signal = load_projection(signal_path)
    af = load_projection(af_path)
    rows = min(signal.shape[0], af.shape[0])
    cols = min(signal.shape[1], af.shape[1])
    if max(abs(signal.shape[0] - af.shape[0]), abs(signal.shape[1] - af.shape[1])) > 1:
        raise ValueError(
            f"channel shapes differ by more than one pixel: {signal.shape} vs {af.shape}"
        )
    return signal[:rows, :cols], af[:rows, :cols]


def save_uint16(path: Path, image: FloatImage, description: str) -> None:
    """Clip to the uint16 range and write, recording the method in ImageDescription."""
    tifffile.imwrite(path, np.clip(image, 0, 65535).astype(np.uint16), description=description)
