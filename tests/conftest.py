"""Synthetic BigStitcher (BigDataViewer HDF5) acquisitions, modelled on real fused output."""

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

VOXEL_SIZE_XYZ = (3.26, 3.26, 5.0)
CHANNEL_IDS = {"638 nm": 0, "488 nm": 1, "561 nm": 2}
# deliberately not the identity, so a positional channel -> setup lookup reads the wrong data
SETUP_OF_CHANNEL = {"638 nm": 2, "488 nm": 0, "561 nm": 1}

Volumes = dict[str, npt.NDArray[np.uint16]]
WriteAcquisition = Callable[..., Path]


def _xml(shape_zyx: tuple[int, ...]) -> str:
    z, y, x = shape_zyx
    vx, vy, vz = VOXEL_SIZE_XYZ
    setups = "".join(
        f"""
      <ViewSetup>
        <id>{SETUP_OF_CHANNEL[name]}</id>
        <name>setup {SETUP_OF_CHANNEL[name]}</name>
        <size>{x} {y} {z}</size>
        <voxelSize><unit>um</unit><size>{vx} {vy} {vz}</size></voxelSize>
        <attributes>
          <illumination>0</illumination><channel>{cid}</channel><tile>0</tile><angle>0</angle>
        </attributes>
      </ViewSetup>"""
        for name, cid in CHANNEL_IDS.items()
    )
    channels = "".join(
        f"<Channel><id>{cid}</id><name>{name}</name></Channel>" for name, cid in CHANNEL_IDS.items()
    )
    # stale entries carried over from the pre-fusion dataset, as in real BigStitcher output
    missing = "".join(f'<MissingView timepoint="0" setup="{s}" />' for s in (2, 3, 6, 7))
    return f"""<?xml version='1.0' encoding='utf-8'?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5"><hdf5 type="relative">stitched.h5</hdf5></ImageLoader>
    <ViewSetups>{setups}
      <Attributes name="channel">{channels}</Attributes>
    </ViewSetups>
    <MissingViews>{missing}</MissingViews>
  </SequenceDescription>
</SpimData>
"""


def write_bdv(
    h5_path: Path,
    volumes: Volumes,
    chunks: tuple[int, int, int] = (8, 64, 64),
    extra_level: bool = False,
) -> Path:
    """Write volumes as BigDataViewer HDF5, i.e. uint16 data stored as int16."""
    shapes = {v.shape for v in volumes.values()}
    assert len(shapes) == 1
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        for name, volume in volumes.items():
            setup = f"s{SETUP_OF_CHANNEL[name]:02d}"
            levels = [volume, volume[:, ::2, ::2]] if extra_level else [volume]
            resolutions = [[1, 1, 1], [2, 2, 1]][: len(levels)]  # x, y, z
            f.create_dataset(f"{setup}/resolutions", data=np.array(resolutions, dtype=np.float64))
            f.create_dataset(f"{setup}/subdivisions", data=np.array([chunks[::-1]] * len(levels)))
            for level, data in enumerate(levels):
                f.create_dataset(
                    f"t00000/{setup}/{level}/cells",
                    data=data.view(np.int16),
                    chunks=tuple(min(c, s) for c, s in zip(chunks, data.shape)),
                )
    h5_path.with_suffix(".xml").write_text(_xml(shapes.pop()))
    return h5_path


@pytest.fixture
def write_acquisition(tmp_path: Path) -> WriteAcquisition:
    def write(
        volumes: Volumes,
        date: str = "2026-05-18",
        mouse: str = "N027",
        imaging: str | None = "001",
        **kwargs: object,
    ) -> Path:
        directory = tmp_path / date / mouse
        if imaging is not None:
            directory = directory / imaging
        return write_bdv(directory / "stitched.h5", volumes, **kwargs)  # type: ignore[arg-type]

    return write
