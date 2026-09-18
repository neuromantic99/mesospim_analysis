"""Loading BigStitcher HDF5 output (BigDataViewer format: stitched.h5 + stitched.xml)."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt

Channel = Literal["488 nm", "561 nm", "638 nm"]

FIRST_TIMEPOINT = "t00000"


def get_wavelengths_of_channels(xml_path: Path) -> dict[str, int]:
    """Map channel name (e.g. "638 nm") to BigStitcher channel id."""
    root = ET.parse(xml_path).getroot()
    lookup: dict[str, int] = {}
    for channel in root.iter("Channel"):
        name = channel.findtext("name")
        channel_id = channel.findtext("id")
        if name is None or channel_id is None:
            raise ValueError(f"{xml_path}: <Channel> missing <name> or <id>")
        lookup[name] = int(channel_id)
    return lookup


def xml_path_for(h5_path: Path) -> Path:
    # BigStitcher gives the xml the same name as the h5
    return h5_path.with_suffix(".xml")


@dataclass(frozen=True)
class StitchedMetadata:
    h5_path: Path
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx_um: tuple[float, float, float]
    """Voxel size at pyramid level 0."""
    channel_setups: dict[str, int]
    """Channel name -> ViewSetup id, i.e. which s?? group in the HDF5 holds that channel."""

    def setup_id(self, channel: str) -> int:
        if channel not in self.channel_setups:
            raise KeyError(f"channel {channel!r} not in {sorted(self.channel_setups)}")
        return self.channel_setups[channel]


def read_stitched_metadata(h5_path: Path) -> StitchedMetadata:
    """Parse volume shape, voxel size and the channel -> setup mapping from the BigStitcher xml.

    Setups are matched to channels through each ViewSetup's <channel> attribute rather than
    by position, because setup id only equals channel id when there is exactly one setup per
    channel (true for fused output, not for raw tiled/multi-illumination data).
    """
    xml_path = xml_path_for(h5_path)
    root = ET.parse(xml_path).getroot()
    channel_ids = get_wavelengths_of_channels(xml_path)

    setups_by_channel: dict[int, list[int]] = {}
    shapes: set[tuple[int, int, int]] = set()
    voxel_sizes: set[tuple[float, float, float]] = set()
    for setup in root.iter("ViewSetup"):
        setup_id = setup.findtext("id")
        channel_id = setup.findtext("attributes/channel")
        size_xyz = setup.findtext("size")
        voxel_xyz = setup.findtext("voxelSize/size")
        if setup_id is None or channel_id is None or size_xyz is None or voxel_xyz is None:
            raise ValueError(f"{xml_path}: <ViewSetup> missing id, channel, size or voxelSize")
        setups_by_channel.setdefault(int(channel_id), []).append(int(setup_id))
        x, y, z = (int(v) for v in size_xyz.split())
        shapes.add((z, y, x))
        vx, vy, vz = (float(v) for v in voxel_xyz.split())
        voxel_sizes.add((vz, vy, vx))

    if len(shapes) != 1 or len(voxel_sizes) != 1:
        raise ValueError(f"{xml_path}: setups differ in size or voxel size: {shapes}, {voxel_sizes}")
    voxel_size = voxel_sizes.pop()
    if not np.isclose(voxel_size[1], voxel_size[2], rtol=0.01):
        raise ValueError(f"{xml_path}: anisotropic xy voxel size {voxel_size} is not supported")

    channel_setups: dict[str, int] = {}
    for name, id_of_channel in channel_ids.items():
        setups = setups_by_channel.get(id_of_channel, [])
        if len(setups) != 1:
            raise ValueError(
                f"{xml_path}: channel {name!r} has setups {setups}; expected exactly one "
                "(is this fused BigStitcher output?)"
            )
        channel_setups[name] = setups[0]

    return StitchedMetadata(
        h5_path=h5_path,
        shape_zyx=shapes.pop(),
        voxel_size_zyx_um=voxel_size,
        channel_setups=channel_setups,
    )


def pyramid_downsampling(h5_file: h5py.File, setup_id: int) -> list[tuple[int, int, int]]:
    """Downsampling factor of each pyramid level, as (z, y, x). BDV stores these as x, y, z."""
    resolutions = np.asarray(h5_file[f"s{setup_id:02d}"]["resolutions"])
    return [(int(z), int(y), int(x)) for x, y, z in resolutions]


def open_channel_dataset(h5_file: h5py.File, setup_id: int, pyramid_level: int) -> h5py.Dataset:
    """The (z, y, x) dataset for one channel at one pyramid level."""
    return h5_file[FIRST_TIMEPOINT][f"s{setup_id:02d}"][str(pyramid_level)]["cells"]


def as_unsigned(block: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """BigDataViewer stores uint16 as int16 (Java has no unsigned short); undo the wrap.

    Without this, pixels above 32767 read as negative and vanish from a max projection.
    """
    return block.view(np.uint16) if block.dtype == np.int16 else block


def load_bigstitched_data(
    path: Path,
    pyramid_level: int,
    channel: Channel,
    stack_start: int,
    stack_end: int,
) -> da.Array:
    """Lazily load a z-range of one channel, e.g. for viewing in napari."""
    setup_id = read_stitched_metadata(path).setup_id(channel)

    # Deliberately not using a context manager: the file must stay open for
    # the lifetime of the returned dask array so chunks can be read lazily
    # (e.g. by napari, only for what's actually rendered) instead of eagerly
    # reading everything into memory here.
    f = h5py.File(path, "r")
    dataset = open_channel_dataset(f, setup_id, pyramid_level)

    # dask.array is unannotated, so pin each result to da.Array
    array: da.Array = da.from_array(dataset, chunks=dataset.chunks)  # type: ignore[no-untyped-call]
    if array.dtype == np.int16:
        array = array.view(np.uint16)  # type: ignore[no-untyped-call]
    stack: da.Array = array[stack_start:stack_end, :, :]
    return stack
