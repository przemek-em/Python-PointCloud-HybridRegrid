"""Opt-in desktop check: python tests/ui_smoke.py (writes diagnostics/)."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import time
import tkinter as tk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import RegridApp
from cloud_io import read_cloud


def wait(root, app):
    deadline = time.monotonic() + 30
    while app.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    assert not app.busy, "GUI operation timed out"


def main():
    root = tk.Tk()
    app = RegridApp(root)
    errors = []
    diagnostics = ROOT / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    try:
        with TemporaryDirectory() as directory, patch("app.messagebox.showerror", side_effect=lambda title, message, **kw: errors.append((title, message))):
            folder = Path(directory)
            source, output = folder / "scan.xyz", folder / "regridded.laz"
            np.savetxt(source, [[x + .2, y + .2, .2] for x in range(3) for y in range(3) if (x, y) != (1, 1)])
            app.input_path.set(str(source))
            app.output_path.set(str(output))
            app.auto.set(False)
            app.cell_size.set("1")
            app.iterations.set("1")
            app._sync_controls()
            root.update()
            assert app.voxel_entry.instate(["disabled"])
            app.voxel_matches_cell.set(False)
            app.keep_below.set(False)
            app._sync_controls()
            assert not app.voxel_entry.instate(["disabled"])
            assert app.delta_entry.instate(["disabled"])
            app.voxel_matches_cell.set(True)
            app.keep_below.set(True)
            app._sync_controls()
            app.estimate()
            wait(root, app)
            assert float(app.cell_size.get()) > 0
            app.cell_size.set("1")
            app.run()
            wait(root, app)
            assert not errors, errors
            result = read_cloud(output)
            assert len(result.xyz) == 9 and result.generated.sum() == 1
            assert "Saved 9 points" in app.status.get()
            assert "Report:" in app.log.get("1.0", "end")
            assert app.run_button.winfo_y() + app.run_button.winfo_height() <= app.run_button.master.winfo_height()
            app.input_path.set("example_scan.laz")
            app.output_path.set("example_scan_regrid.laz")
            root.update()
            # Check layout from widget geometry without capturing the desktop.
            root.update()
            for widget in [app.run_button, app.cancel_button, app.size_entry, app.voxel_entry, app.height_box, app.delta_entry]:
                assert widget.winfo_ismapped(), str(widget)
                assert widget.winfo_rooty() >= root.winfo_rooty()
                assert widget.winfo_rooty() + widget.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            # The new mode fills a truly vertical point-cloud surface.
            wall = folder / "wall.xyz"
            wall_xyz = np.array([[.2, x+.2, y+.2] for x in range(-5, 6) for y in range(-5, 6) if not (abs(x) <= 1 and abs(y) <= 1)])
            np.savetxt(wall, wall_xyz)
            app.input_path.set(str(wall))
            app.mode.set("3D surface gap filling")
            app._mode_changed()
            root.update()
            assert app.surface_frame.winfo_ismapped()
            assert not app.heightfield_frame.winfo_ismapped()
            assert app.surface_spacing_entry.instate(["disabled"])
            assert app.surface_fit.get().startswith("Boundary")
            assert all(control.instate(["disabled"]) for control in app.surface_local_controls)
            app.surface_fit.set("Quadratic (curved local patch)")
            app._sync_controls()
            assert all(not control.instate(["disabled"]) for control in app.surface_local_controls)
            app.surface_fit.set("Boundary patches (bounded holes)")
            app._sync_controls()
            app.estimate()
            wait(root, app)
            assert float(app.surface_spacing.get()) > 0
            app.surface_auto.set(False)
            app.surface_spacing.set("1")
            app.surface_workers.set("2")
            assert app.surface_workers_box.instate(["readonly"])
            app._sync_controls()
            filled_wall = folder / "wall_filled.laz"
            app.output_path.set(str(filled_wall))
            app.run()
            wait(root, app)
            assert not errors, errors
            filled = read_cloud(filled_wall)
            surface_report = json.loads(Path(str(filled_wall)+".regrid.json").read_text())
            assert surface_report["parameters"]["method"] == "boundary"
            assert surface_report["boundary_patches"]["added_output_cells"] > 0
            assert len(filled.xyz) > len(wall_xyz)
            assert filled.generated.sum() == len(filled.xyz)-len(wall_xyz)
            np.testing.assert_allclose(filled.xyz[:, 0], .2, atol=.001)
            for widget in [app.run_button, app.surface_spacing_entry, app.surface_gap_entry, app.surface_fit_box, app.surface_workers_box]:
                assert widget.winfo_ismapped()
                assert widget.winfo_rooty()+widget.winfo_height() <= root.winfo_rooty()+root.winfo_height()
            app.mode.set("Rebuild uniform 3D cloud")
            app._mode_changed()
            root.update()
            assert app.rebuild_frame.winfo_ismapped()
            assert not app.surface_frame.winfo_ismapped()
            assert app.rebuild_spacing_entry.instate(["disabled"])
            assert app.rebuild_reach_entry.instate(["disabled"])
            app.estimate()
            wait(root, app)
            assert float(app.rebuild_spacing.get()) == 1
            app.rebuild_auto.set(False)
            app.rebuild_spacing.set("0.5")
            app.rebuild_workers.set("2")
            app._sync_controls()
            assert app.rebuild_workers_box.instate(["readonly"])
            rebuilt_wall = folder / "wall_uniform.laz"
            app.output_path.set(str(rebuilt_wall))
            app.run()
            wait(root, app)
            assert not errors, errors
            rebuilt = read_cloud(rebuilt_wall)
            report = json.loads(Path(str(rebuilt_wall)+".regrid.json").read_text())
            assert report["mode"] == "uniform_rebuild_3d"
            assert report["parameters"]["completion"] == "strong"
            assert report["boundary_patches"] is not None
            assert len(rebuilt.xyz) > len(wall_xyz)
            local = (rebuilt.xyz-np.array(report["grid_origin"]))/report["resolved_spacing"]-.5
            np.testing.assert_allclose(local, np.rint(local), atol=.002)
            assert rebuilt.generated.sum() == report["added_points"]
            for widget in [app.run_button, app.rebuild_spacing_entry, app.rebuild_positions_box, app.rebuild_completion_box, app.rebuild_workers_box]:
                assert widget.winfo_ismapped()
                assert widget.winfo_rooty()+widget.winfo_height() <= root.winfo_rooty()+root.winfo_height()
            target = folder / "cancelled.xyz"
            app.output_path.set(str(target))
            app.run()
            app.cancel_job()
            wait(root, app)
            assert "Cancelled" in app.status.get(), app.status.get()
            assert not target.exists()
            assert not errors, errors
        report = {"result": "PASS", "checks": ["estimate spacing", "background job", "LAS/LAZ output", "generated marker", "hybrid controls", "3D surface mode and vertical-wall fill", "uniform rebuild mode and exact lattice export", "native spacing estimate", "CPU worker control", "JSON report", "GUI cancellation"]}
        (diagnostics / "gui-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
