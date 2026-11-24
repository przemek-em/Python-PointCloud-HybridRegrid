"""Point-cloud files and transactional output for the Tkinter test tool."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Callable
import json
import os
import tempfile

import numpy as np

from cloud import Cloud, check_cancel

SUPPORTED = {".las", ".laz", ".ply", ".xyz", ".txt"}


def _rgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("RGB contains invalid colour values.")
    # Integer PLY channels have a declared bit depth; float/XYZ channels use
    # [0,1], [0,255] or [0,65535]. A uint16 PLY is always treated as 16 bit.
    if values.dtype == np.uint16:
        divisor = 65535
    elif values.dtype == np.uint8:
        divisor = 255
    else:
        maximum = values.max(initial=0)
        divisor = 1 if maximum <= 1 else 255 if maximum <= 255 else 65535
    return np.clip(values.astype(np.float64) / divisor, 0, 1)


def read_cloud(path: str | Path, cancel: Event | None = None, log: Callable[[str], None] | None = None) -> Cloud:
    path = Path(path)
    log = log or (lambda message: None)
    check_cancel(cancel)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Supported files: LAS, LAZ, PLY and whitespace-separated XYZ/TXT.")
    metadata = {"input_path": str(path.resolve()), "input_format": suffix[1:]}
    if suffix in {".las", ".laz"}:
        import laspy

        with laspy.open(path) as reader:
            if reader.header.point_format.id in {4, 5, 9, 10}:
                raise ValueError("Waveform LAS point formats 4/5/9/10 are unsupported. Export a non-waveform point cloud first.")
            log(f"Reading {reader.header.point_count:,} points ({suffix[1:].upper()})")
            data = reader.read()
        check_cancel(cancel)
        dimensions = set(data.point_format.dimension_names)
        extra = "regrid_generated" in dimensions
        rgb = np.column_stack((data.red, data.green, data.blue)).astype(np.float64) / 65535 if "red" in dimensions else None
        metadata["las_data"] = data
        metadata["retained_metadata"] = "LAS CRS/VLRs and original point format. Other point fields use a representative input record; added records start at zero. COPC indexes are removed."
        log("Keeping LAS header, CRS records, colour, intensity and classification.")
        return Cloud(np.column_stack((data.x, data.y, data.z)), rgb,
                     np.asarray(data.intensity, dtype=np.float64), np.asarray(data.classification).copy(),
                     np.asarray(data.regrid_generated, dtype=bool) if extra else np.asarray(data.synthetic, dtype=bool), metadata=metadata)
    if suffix == ".ply":
        from ply_io import read_vertices

        vertex, has_faces = read_vertices(path)
        names = set(vertex.dtype.names or ())
        if not {"x", "y", "z"} <= names:
            raise ValueError("PLY vertices need x, y and z properties.")
        rgb = _rgb(np.column_stack([vertex[c] for c in ("red", "green", "blue")])) if {"red", "green", "blue"} <= names else None
        intensity = vertex["intensity"].astype(np.float64) if "intensity" in names else None
        classification = vertex["classification"].copy() if "classification" in names else None
        generated = vertex["regrid_generated"].astype(bool) if "regrid_generated" in names else None
        unused = names - {"x", "y", "z", "red", "green", "blue", "intensity", "classification", "regrid_generated"}
        if unused:
            log("PLY properties omitted: " + ", ".join(sorted(unused)))
        if has_faces:
            log("PLY mesh: using vertices only; faces are omitted.")
        metadata["omitted_ply_properties"] = sorted(unused)
        check_cancel(cancel)
        return Cloud(np.column_stack((vertex["x"], vertex["y"], vertex["z"])), rgb, intensity, classification, generated, metadata=metadata)
    values = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if values.shape[1] not in (3, 6):
        raise ValueError("XYZ/TXT needs 3 columns (X Y Z) or 6 columns (X Y Z R G B). Use whitespace, with # for comments.")
    check_cancel(cancel)
    return Cloud(values[:, :3], _rgb(values[:, 3:]) if values.shape[1] == 6 else None, metadata=metadata)


def _write_las(path: Path, cloud: Cloud, voxel_size: float, compressed: bool, cancel: Event | None, progress: Callable[[float], None]) -> None:
    import laspy

    source = cloud.metadata.get("las_data")
    needs_extended_classes = cloud.classification is not None and np.max(cloud.classification) > 31
    if source is not None:
        header = deepcopy(source.header)
    elif needs_extended_classes:
        header = laspy.LasHeader(point_format=7 if cloud.rgb is not None else 6, version="1.4")
    else:
        header = laspy.LasHeader(point_format=3 if cloud.rgb is not None else 1, version="1.2")
    # A rewritten regular LAS/LAZ cannot retain the spatial offsets in a COPC index.
    header.vlrs = laspy.vlrs.vlrlist.VLRList(v for v in header.vlrs if v.user_id.lower() not in {"copc", "laszip encoded"})
    if header.evlrs is not None:
        header.evlrs = laspy.vlrs.vlrlist.VLRList(v for v in header.evlrs if v.user_id.lower() != "copc")
    if "regrid_generated" not in set(header.point_format.dimension_names):
        header.add_extra_dim(laspy.ExtraBytesParams(name="regrid_generated", type=np.uint8, description="1 = estimated / generated point"))
    elif np.dtype(header.point_format.dimension_by_name("regrid_generated").dtype) != np.dtype(np.uint8):
        raise ValueError("The input's regrid_generated field has an incompatible type.")
    minimum, maximum = cloud.xyz.min(axis=0), cloud.xyz.max(axis=0)
    # Keep at least the input precision, and enough extra precision to place
    # centroids. Widen the scale only if the LAS signed integer range requires it.
    scales = np.minimum(header.scales, voxel_size / 1000)
    header.scales = np.maximum(scales, (maximum - minimum) / (2**32 - 4))
    header.offsets = minimum + (maximum - minimum) / 2
    header.generating_software = "Hybrid Regrid Tk 0.6"
    with laspy.open(path, mode="w", header=header, do_compress=compressed) as writer:
        for start in range(0, len(cloud.xyz), 65_536):
            check_cancel(cancel)
            section = slice(start, start + 65_536)
            xyz = cloud.xyz[section]
            records = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
            indices = cloud.source_indices[section]
            valid = indices >= 0
            if source is not None and valid.any():
                # Copy raw fields, including packed return/scan flags and extra
                # dimensions, then override the resampled attributes below.
                for name in source.points.array.dtype.names:
                    if name in records.array.dtype.names:
                        records.array[name][valid] = source.points.array[name][indices[valid]]
            records.x, records.y, records.z = xyz.T
            if cloud.rgb is not None and "red" in set(header.point_format.dimension_names):
                records.red, records.green, records.blue = np.rint(cloud.rgb[section] * 65535).astype(np.uint16).T
            if cloud.intensity is not None:
                records.intensity = np.rint(np.clip(cloud.intensity[section], 0, 65535)).astype(np.uint16)
            if cloud.classification is not None:
                records.classification = cloud.classification[section]
            records.regrid_generated = cloud.generated[section].astype(np.uint8)
            records.synthetic = np.asarray(records.synthetic) | cloud.generated[section]
            writer.write_points(records)
            progress(min(1.0, (start + len(xyz)) / len(cloud.xyz)))
        if header.evlrs:
            writer.write_evlrs(header.evlrs)


def _write_ply(path: Path, cloud: Cloud, cancel: Event | None) -> None:
    from ply_io import write_vertices

    dtype = [(axis, "f8") for axis in ("x", "y", "z")]
    if cloud.rgb is not None:
        dtype += [(channel, "u1") for channel in ("red", "green", "blue")]
    if cloud.intensity is not None:
        dtype.append(("intensity", "f8"))
    if cloud.classification is not None:
        dtype.append(("classification", "u1"))
    dtype.append(("regrid_generated", "u1"))
    vertex = np.empty(len(cloud.xyz), dtype=dtype)
    for i, axis in enumerate(("x", "y", "z")):
        vertex[axis] = cloud.xyz[:, i]
    if cloud.rgb is not None:
        for i, channel in enumerate(("red", "green", "blue")):
            vertex[channel] = np.rint(cloud.rgb[:, i] * 255).astype(np.uint8)
    for name in ("intensity", "classification"):
        if getattr(cloud, name) is not None:
            vertex[name] = getattr(cloud, name)
    vertex["regrid_generated"] = cloud.generated.astype(np.uint8)
    check_cancel(cancel)
    write_vertices(path, vertex)


def write_cloud(path: str | Path, cloud: Cloud, voxel_size: float, cancel: Event | None = None, progress: Callable[[float], None] | None = None, overwrite: bool = False) -> None:
    """Write to a temporary sibling and commit only after successful completion."""
    path = Path(path).resolve()
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError("Output extension must be .las, .laz, .ply, .xyz or .txt.")
    original = cloud.metadata.get("input_path")
    if original and Path(original).resolve() == path:
        raise ValueError("Choose an output path different from the input file.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    if not path.parent.is_dir():
        raise ValueError("The output folder does not exist.")
    check_cancel(cancel)
    progress = progress or (lambda fraction: None)
    handle, temporary = tempfile.mkstemp(prefix=".hybrid-regrid-", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(temporary)
    try:
        suffix = path.suffix.lower()
        if suffix in {".las", ".laz"}:
            _write_las(temporary, cloud, voxel_size, suffix == ".laz", cancel, progress)
        elif suffix == ".ply":
            _write_ply(temporary, cloud, cancel)
        else:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("# X Y Z" + (" R G B (0..255)" if cloud.rgb is not None else "") + "\n")
                for start in range(0, len(cloud.xyz), 65_536):
                    check_cancel(cancel)
                    xyz = cloud.xyz[start:start + 65_536]
                    values = xyz if cloud.rgb is None else np.column_stack((xyz, np.rint(cloud.rgb[start:start + 65_536] * 255)))
                    np.savetxt(stream, values, fmt="%.17g")
                    progress(min(1, (start + len(xyz)) / len(cloud.xyz)))
        check_cancel(cancel)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-overwrite commit, including when another process creates
            # the requested output while this job is still running.
            os.link(temporary, path)
            temporary.unlink()
        progress(1.0)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=".regrid-report-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
