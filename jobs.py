"""Shared processing job, kept independent of Tk so it is testable."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import perf_counter

from cloud_io import read_cloud, write_cloud, write_report
from cloud import check_cancel
from hybrid import Parameters, regrid
from surface_fill import Parameters as SurfaceParameters, fill_gaps
from uniform_rebuild import Parameters as RebuildParameters, rebuild


def process_file(input_path, output_path, params: Parameters, cancel: Event, progress, log, overwrite=False) -> dict:
    start = perf_counter()
    progress(0.01, "Reading point cloud")
    cloud = read_cloud(input_path, cancel, log)
    check_cancel(cancel)
    log(f"Input: {len(cloud.xyz):,} points")
    log("Bounds: " + ", ".join(f"{a:.8g} to {b:.8g}" for a, b in zip(cloud.xyz.min(axis=0), cloud.xyz.max(axis=0))))
    result = regrid(cloud, params, cancel, lambda f, message: progress(0.15 + f * 0.65, message))
    log(f"3D voxel size: {result.voxel_size:.10g} input units")
    log(f"XY cell size: {result.cell_size:.10g}; heightfield cells: {result.heightfield_count:,}; added HF cells: {result.added_count:,}")
    log(f"3D voxel candidates: {result.occupied_voxels:,}; retained walls: {result.walls_kept:,}, caves: {result.caves_kept:,}, fringe: {result.fringe_kept:,}")
    if Path(output_path).suffix.lower() in {".xyz", ".txt"}:
        log("XYZ export keeps positions and RGB only. Use LAS/LAZ or PLY to retain generated-point markers and other supported attributes.")
    progress(0.81, "Writing output; input file is kept intact")
    write_cloud(output_path, result.cloud, min(result.cell_size, result.voxel_size), cancel, lambda f: progress(0.81 + 0.18 * f, "Writing output"), overwrite)
    report = {
        "tool": "Hybrid Regrid Tk 0.6", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(input_path).resolve()), "output": str(Path(output_path).resolve()),
        "parameters": asdict(params), "resolved_voxel_size": result.voxel_size, "resolved_cell_size": result.cell_size,
        "coordinate_units": "Same as input; no coordinate reprojection or unit conversion",
        "input_points": result.input_count, "occupied_voxels": result.occupied_voxels,
        "original_heightfield_cells": result.heightfield_count,
        "heightfield_output_points": result.heightfield_count + result.added_count,
        "voxel_wall_points": result.walls_kept, "voxel_cave_points": result.caves_kept, "voxel_fringe_points": result.fringe_kept,
        "added_points": result.added_count, "output_points": len(result.cloud.xyz),
        "marked_generated_points": int(result.cloud.generated.sum()), "completed_fill_passes": result.passes,
        "elapsed_seconds": round(perf_counter() - start, 3),
        "bounds_min": result.cloud.xyz.min(axis=0).tolist(), "bounds_max": result.cloud.xyz.max(axis=0).tolist(),
        "heightfield_origin": result.heightfield_origin.tolist(), "voxel_origin": result.voxel_origin.tolist(),
        "heightfield_statistic_note": "Median uses the upper middle sample for even counts, matching the editor.",
        "port_differences": "Float64 coordinates, wide integer keys, deterministic class ties, and propagation of fill colours/intensity across passes.",
        "limitations": "Hybrid uses Z-up heightfield slopes, not normals. Missing geometry is estimated. Isolated walls or upper sheets can be omitted; output layers may overlap. No ground-truth accuracy is certified.",
        "input_metadata_notes": {k: v for k, v in cloud.metadata.items() if k not in {"las_data", "input_path"}},
    }
    report_path = Path(str(output_path) + ".regrid.json")
    try:
        write_report(report_path, report)
        report["report_path"] = str(report_path)
        log(f"Report: {report_path}")
    except OSError as error:
        report["report_error"] = str(error)
        log(f"Output saved, but report could not be written: {error}")
    log(f"Saved {len(result.cloud.xyz):,} points in {report['elapsed_seconds']:.2f} s")
    progress(1, "Completed")
    return report


def process_surface_file(input_path, output_path, params: SurfaceParameters, cancel: Event, progress, log, overwrite=False) -> dict:
    start = perf_counter()
    progress(.01, "Reading 3D point cloud")
    cloud = read_cloud(input_path, cancel, log)
    log(f"3D surface fill: keeping all {len(cloud.xyz):,} original points")
    result = fill_gaps(cloud, params, cancel, lambda f, message: progress(.12 + .68*f, message))
    log(f"Support points: {result.support_points:,}; reliable local directions: {result.reliable_normals:,}")
    log(f"CPU worker limit: {result.workers}; nested BLAS threads: 1")
    log(f"Spacing: {result.spacing:.10g}; maximum distance to measured support: {result.max_gap_distance:.10g}")
    log(f"Added estimates: {result.added_count:,}; original points retained: {result.input_count:,}")
    if result.boundary_patches is not None:
        log(f"Boundary patches: {result.boundary_patches['patched_holes']:,} chart patches; {result.boundary_patches['added_output_cells']:,} new output cells")
    if Path(output_path).suffix.lower() in {".xyz", ".txt"}:
        log("XYZ output has no per-point generated markers. Use LAS/LAZ or PLY to retain them.")
    progress(.81, "Writing original points and estimated additions")
    write_cloud(output_path, result.cloud, result.spacing, cancel, lambda f: progress(.81 + .18*f, "Writing output"), overwrite)
    report = {
        "tool": "Hybrid Regrid Tk 0.6", "mode": "surface_fill_3d",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(input_path).resolve()), "output": str(Path(output_path).resolve()),
        "parameters": asdict(params), "resolved_spacing": result.spacing,
        "resolved_max_gap_distance": result.max_gap_distance,
        "input_points": result.input_count, "output_points": len(result.cloud.xyz),
        "added_points": result.added_count, "marked_generated_points": int(result.cloud.generated.sum()),
        "completed_fill_passes": result.passes, "support_points": result.support_points,
        "reliable_normals": result.reliable_normals, "tested_candidates": result.tested_candidates,
        "worker_limit": result.workers, "blas_threads_per_worker": 1,
        "boundary_patches": result.boundary_patches,
        "elapsed_seconds": round(perf_counter()-start, 3),
        "coordinate_units": "Same as input; orientation-independent local surface fits; no reprojection",
        "limitations": "Interpolation estimates missing geometry. Boundary patches infer bounded holes and can also close genuine openings. Nearby sheets or ambiguous regions may be unresolved; oversize/reach-limited patches can be skipped. Original points are kept; added points never support another fit.",
        "input_metadata_notes": {k: v for k, v in cloud.metadata.items() if k not in {"las_data", "input_path"}},
    }
    report_path = Path(str(output_path)+".regrid.json")
    try:
        write_report(report_path, report)
        report["report_path"] = str(report_path)
        log(f"Report: {report_path}")
    except OSError as error:
        report["report_error"] = str(error)
        log(f"Output saved, but report could not be written: {error}")
    log(f"Saved {len(result.cloud.xyz):,} points in {report['elapsed_seconds']:.2f} s")
    progress(1, "Completed")
    return report


def process_rebuild_file(input_path, output_path, params: RebuildParameters, cancel: Event, progress, log, overwrite=False) -> dict:
    start = perf_counter()
    progress(.01, "Reading point cloud for uniform rebuilding")
    cloud = read_cloud(input_path, cancel, log)
    log(f"Rebuilding a new cloud from {len(cloud.xyz):,} input points")
    result = rebuild(cloud, params, cancel, lambda f, m: progress(.1+.7*f, m))
    log(f"Native input spacing: {result.native_spacing:.10g}; output grid spacing: {result.spacing:.10g}")
    log(f"Fill reach: {result.reach:.10g}; completion: {params.completion}; CPU worker limit: {result.workers}")
    log(f"Observed grid cells: {result.occupied_cells:,}; reconstructed cells: {result.added_count:,}; output: {len(result.cloud.xyz):,}")
    log(f"Stop condition: {result.stop_reason}")
    if result.boundary_patches is not None:
        log(f"Boundary patches: {result.boundary_patches['patched_holes']:,} chart patches; {result.boundary_patches['added_output_cells']:,} new output cells")
    progress(.81, "Writing rebuilt point cloud")
    write_cloud(output_path, result.cloud, result.spacing, cancel, lambda f: progress(.81+.18*f, "Writing rebuilt cloud"), overwrite)
    report = {
        "tool": "Hybrid Regrid Tk 0.6", "mode": "uniform_rebuild_3d",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(input_path).resolve()), "output": str(Path(output_path).resolve()),
        "parameters": asdict(params), "resolved_spacing": result.spacing,
        "native_input_spacing": result.native_spacing, "resolved_reach": result.reach,
        "grid_origin": result.origin.tolist(), "input_points": result.input_count,
        "occupied_input_cells": result.occupied_cells, "added_points": result.added_count,
        "output_points": len(result.cloud.xyz), "marked_generated_points": int(result.cloud.generated.sum()),
        "completed_rounds": result.rounds, "tested_candidates": result.tested_candidates,
        "reliable_supports": result.reliable_supports, "worker_limit": result.workers,
        "boundary_patches": result.boundary_patches,
        "stop_reason": result.stop_reason, "elapsed_seconds": round(perf_counter()-start, 3),
        "reference_input_generated_points": int(cloud.generated.sum()),
        "coordinate_units": "Same as input, one point per occupied surface voxel; no reprojection",
        "limitations": "A resampled surface cloud, not a solid volume. Exact grid centres can move samples off the fitted surface. Completion infers missing geometry from the fixed input and can bridge real openings or nearby surfaces, especially in Strong mode. Unsupported areas can remain empty.",
    }
    report_path = Path(str(output_path)+".regrid.json")
    try:
        write_report(report_path, report)
        report["report_path"] = str(report_path)
        log(f"Report: {report_path}")
    except OSError as error:
        report["report_error"] = str(error)
        log(f"Output saved, but report could not be written: {error}")
    log(f"Saved new cloud in {report['elapsed_seconds']:.2f} s")
    progress(1, "Completed")
    return report
