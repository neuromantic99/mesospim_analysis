"""Max projections of z-slabs, read from disk in chunk-sized blocks."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from mesospim_analysis.types import FloatImage
from mesospim_analysis.utils import as_unsigned

MAX_READ_BLOCK_PLANES = 64


class Volume(Protocol):
    """Anything indexable like a (z, y, x) array: an h5py Dataset or a numpy array."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key: slice) -> npt.NDArray[np.generic]: ...


@dataclass(frozen=True)
class Slab:
    index: int
    z_start: int
    z_end: int
    """Exclusive."""
    projection: FloatImage


def planes_per_slab(thickness_um: float, z_step_um: float) -> int:
    planes = round(thickness_um / z_step_um)
    if planes < 1:
        raise ValueError(f"slab of {thickness_um} um is thinner than one {z_step_um} um plane")
    return planes


def iter_slab_projections(
    volume: Volume,
    planes: int,
    z_start: int = 0,
    z_end: int | None = None,
    read_block: int = 32,
) -> Iterator[Slab]:
    """Yield the max projection of each consecutive `planes`-thick slab in [z_start, z_end).

    Reads `read_block` planes at a time and keeps running maxima, so each HDF5 chunk is
    read once however slabs and chunks align. Set `read_block` to the dataset's z chunk
    size. A trailing partial slab is dropped so every projection spans the same depth.
    """
    z_end = volume.shape[0] if z_end is None else min(z_end, volume.shape[0])
    n_slabs = (z_end - z_start) // planes
    if n_slabs < 1:
        raise ValueError(f"z range [{z_start}, {z_end}) is shorter than one slab of {planes} planes")
    stop = z_start + n_slabs * planes
    read_block = max(1, min(read_block, MAX_READ_BLOCK_PLANES))

    current: FloatImage | None = None
    for block_start in range(z_start, stop, read_block):
        block = as_unsigned(np.asarray(volume[block_start : min(block_start + read_block, stop)]))
        for offset, plane in enumerate(block):
            z = block_start + offset
            plane_f = plane.astype(np.float32)
            current = plane_f if current is None else np.maximum(current, plane_f)
            if (z - z_start + 1) % planes == 0:
                index = (z - z_start) // planes
                yield Slab(index, z - planes + 1, z + 1, current)
                current = None
