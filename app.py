"""Run this module to open the standalone hybrid regrid window."""
from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import os
import sys
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from cloud_io import SUPPORTED, read_cloud
from jobs import process_file, process_surface_file, process_rebuild_file
from cloud import Cancelled, auto_voxel_size
from hybrid import Parameters
from surface_fill import Parameters as SurfaceParameters
from cpu_workers import resolve_workers
from uniform_rebuild import Parameters as RebuildParameters, estimate_native_spacing

ROOT = Path(__file__).resolve().parent
FILE_TYPES = [("Point clouds", "*.las *.laz *.ply *.xyz *.txt"), ("LAS / LAZ", "*.las *.laz"), ("PLY", "*.ply"), ("XYZ text", "*.xyz *.txt")]


class RegridApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events = Queue()
        self.cancel = Event()
        self.busy = False
        self.closing = False
        self.output_directory: Path | None = None
        self.controls = []
        self.mode = tk.StringVar(value="Hybrid Regrid")
        self.subtitle = tk.StringVar(value="Heightfield terrain with 3D wall and cave detail")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.auto = tk.BooleanVar(value=True)
        self.cell_size = tk.StringVar(value="0.1")
        self.voxel_size = tk.StringVar(value="0.1")
        self.voxel_matches_cell = tk.BooleanVar(value=True)
        self.height_stat = tk.StringVar(value="Median (upper middle sample)")
        self.iterations = tk.StringVar(value="1")
        self.wall_slope = tk.StringVar(value="45")
        self.keep_below = tk.BooleanVar(value=True)
        self.below_delta = tk.StringVar(value="1.5")
        self.max_added = tk.StringVar(value="200000")
        self.surface_auto = tk.BooleanVar(value=True)
        self.surface_spacing = tk.StringVar(value="0.1")
        self.surface_gap_auto = tk.BooleanVar(value=True)
        self.surface_gap = tk.StringVar(value="0.3")
        self.surface_passes = tk.StringVar(value="3")
        self.surface_angle = tk.StringVar(value="35")
        self.surface_layer = tk.StringVar(value="0.5")
        self.surface_fit = tk.StringVar(value="Boundary patches (bounded holes)")
        self.surface_max_added = tk.StringVar(value="100000")
        self.surface_workers = tk.StringVar(value="Auto")
        self.rebuild_auto = tk.BooleanVar(value=True)
        self.rebuild_spacing = tk.StringVar(value="0.1")
        self.rebuild_positions = tk.StringVar(value="Exact grid centres (regular 3D lattice)")
        self.rebuild_completion = tk.StringVar(value="Strong")
        self.rebuild_reach_auto = tk.BooleanVar(value=True)
        self.rebuild_reach = tk.StringVar(value="1")
        self.rebuild_workers = tk.StringVar(value="Auto")
        self.rebuild_limit = tk.StringVar(value="2000000")
        self.status = tk.StringVar(value="Choose a point cloud to begin.")
        root.title("Hybrid Regrid — Point Cloud Test Tool")
        root.geometry("940x860")
        root.minsize(880, 830)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.root.after(80, self._poll)

    def _register(self, widget):
        self.controls.append(widget)
        return widget

    def _build(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", foreground="#53616d")
        style.configure("TButton", padding=(10, 5))
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)
        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Hybrid Regrid", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.subtitle, style="Subtitle.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Button(header, text="About / help", command=self.about).grid(row=0, column=1, rowspan=2, sticky="e")
        self.mode_box = self._register(ttk.Combobox(header, textvariable=self.mode,
            values=["Hybrid Regrid", "3D surface gap filling", "Rebuild uniform 3D cloud"], state="readonly"))
        self.mode_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_changed)

        files = ttk.LabelFrame(main, text="Files", padding=12)
        files.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        files.columnconfigure(1, weight=1)
        for row, label, variable, command in [(0, "Input", self.input_path, self.browse_input), (1, "Output", self.output_path, self.browse_output)]:
            ttk.Label(files, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            self._register(ttk.Entry(files, textvariable=variable)).grid(row=row, column=1, sticky="ew", pady=4)
            self._register(ttk.Button(files, text="Browse…", command=command)).grid(row=row, column=2, padx=(8, 0), pady=4)
        ttk.Label(files, text="LAS / LAZ · PLY · XYZ text     Output is saved as a separate file.", style="Subtitle.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(7, 0))

        settings = ttk.LabelFrame(main, text="Heightfield", padding=12)
        self.heightfield_frame = settings
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="XY cell size").grid(row=0, column=0, sticky="w", padx=(0, 16))
        row = ttk.Frame(settings)
        row.grid(row=0, column=1, sticky="ew")
        self.size_entry = self._register(ttk.Entry(row, textvariable=self.cell_size, width=16))
        self.size_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="Automatic", variable=self.auto, command=self._sync_controls)).pack(side="left", padx=10)
        self._register(ttk.Button(row, text="Estimate spacing", command=self.estimate)).pack(side="right")
        ttk.Label(settings, text="Height statistic").grid(row=1, column=0, sticky="w", pady=(9, 0))
        self.height_box = self._register(ttk.Combobox(settings, textvariable=self.height_stat, values=["Median (upper middle sample)", "Mean", "Max (uppermost height)", "Min (lowermost height)"], state="readonly"))
        self.height_box.grid(row=1, column=1, sticky="ew", pady=(9, 0))
        ttk.Label(settings, text="2D hole-fill passes (0 = off)").grid(row=2, column=0, sticky="w", padx=(0, 16), pady=(9, 0))
        self._register(ttk.Spinbox(settings, from_=0, to=8, textvariable=self.iterations, width=8)).grid(row=2, column=1, sticky="w", pady=(9, 0))
        ttk.Label(settings, text="Maximum added HF points").grid(row=3, column=0, sticky="w", pady=(9, 0))
        self._register(ttk.Entry(settings, textvariable=self.max_added, width=16)).grid(row=3, column=1, sticky="w", pady=(9, 0))
        ttk.Label(settings, text="Z is up. Cell and voxel sizes use the input's coordinate units.", style="Subtitle.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(9, 0))

        detail = ttk.LabelFrame(main, text="Retain 3D wall / cave points", padding=12)
        self.detail_frame = detail
        detail.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        detail.columnconfigure(1, weight=1)
        ttk.Label(detail, text="3D voxel size").grid(row=0, column=0, sticky="w", padx=(0, 16))
        row = ttk.Frame(detail)
        row.grid(row=0, column=1, sticky="ew")
        self.voxel_entry = self._register(ttk.Entry(row, textvariable=self.voxel_size, width=16))
        self.voxel_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="Same as XY cell", variable=self.voxel_matches_cell, command=self._sync_controls)).pack(side="left", padx=10)
        ttk.Label(detail, text="Wall slope threshold (degrees)").grid(row=1, column=0, sticky="w", pady=(9, 0))
        self._register(ttk.Spinbox(detail, from_=0, to=89.5, increment=1, textvariable=self.wall_slope, width=8)).grid(row=1, column=1, sticky="w", pady=(9, 0))
        self._register(ttk.Checkbutton(detail, text="Keep points below the heightfield (caves / overhangs)", variable=self.keep_below, command=self._sync_controls)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(9, 4))
        ttk.Label(detail, text="Cave depth threshold (x XY cell)").grid(row=3, column=0, sticky="w")
        self.delta_entry = self._register(ttk.Spinbox(detail, from_=0, to=20, increment=.1, textvariable=self.below_delta, width=8))
        self.delta_entry.grid(row=3, column=1, sticky="w")
        ttk.Label(detail, text="Lower slope thresholds keep more 3D samples. Filled heights are estimates.\nCheck cliffs, stacked surfaces and overhangs against the original scan.", style="Subtitle.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(9, 0))

        surface = self.surface_frame = ttk.LabelFrame(main, text="3D surface gap filling — walls, ground and cave sheets", padding=12)
        surface.grid(row=2, column=0, rowspan=2, sticky="new", pady=(0, 12))
        surface.columnconfigure(1, weight=1)
        ttk.Label(surface, text="Target point spacing").grid(row=0, column=0, sticky="w", padx=(0, 16))
        row = ttk.Frame(surface)
        row.grid(row=0, column=1, sticky="ew")
        self.surface_spacing_entry = self._register(ttk.Entry(row, textvariable=self.surface_spacing, width=14))
        self.surface_spacing_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="Automatic", variable=self.surface_auto, command=self._sync_controls)).pack(side="left", padx=8)
        self._register(ttk.Button(row, text="Estimate spacing", command=self.estimate)).pack(side="right")
        ttk.Label(surface, text="Max. distance to measured point").grid(row=1, column=0, sticky="w", pady=(9, 0))
        row = ttk.Frame(surface)
        row.grid(row=1, column=1, sticky="ew", pady=(9, 0))
        self.surface_gap_entry = self._register(ttk.Entry(row, textvariable=self.surface_gap, width=14))
        self.surface_gap_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="3 x spacing", variable=self.surface_gap_auto, command=self._sync_controls)).pack(side="left", padx=8)
        self.surface_local_controls = []
        for index, label, variable, lower, upper, increment in [
            (2, "Fill passes", self.surface_passes, 1, 8, 1),
            (3, "Normal angle tolerance (degrees)", self.surface_angle, 5, 75, 1),
            (4, "Layer tolerance (x spacing)", self.surface_layer, .05, 2, .05)]:
            ttk.Label(surface, text=label).grid(row=index, column=0, sticky="w", pady=(9, 0))
            control = self._register(ttk.Spinbox(surface, from_=lower, to=upper, increment=increment, textvariable=variable, width=12))
            control.grid(row=index, column=1, sticky="w", pady=(9, 0))
            self.surface_local_controls.append(control)
        ttk.Label(surface, text="Interpolation").grid(row=5, column=0, sticky="w", pady=(9, 0))
        self.surface_fit_box = self._register(ttk.Combobox(surface, textvariable=self.surface_fit,
            values=["Boundary patches (bounded holes)", "Quadratic (curved local patch)", "Plane (flat local patch)"], state="readonly"))
        self.surface_fit_box.grid(row=5, column=1, sticky="ew", pady=(9, 0))
        self.surface_fit_box.bind("<<ComboboxSelected>>",self._sync_controls)
        ttk.Label(surface, text="Maximum added points").grid(row=6, column=0, sticky="w", pady=(9, 0))
        self._register(ttk.Entry(surface, textvariable=self.surface_max_added, width=14)).grid(row=6, column=1, sticky="w", pady=(9, 0))
        ttk.Label(surface, text="CPU workers").grid(row=7, column=0, sticky="w", pady=(9, 0))
        self.surface_workers_box = self._register(ttk.Combobox(surface, textvariable=self.surface_workers,
            values=["Auto", "1", "2", "4", "8", "16", "32"], state="readonly", width=12))
        self.surface_workers_box.grid(row=7, column=1, sticky="w", pady=(9, 0))
        ttk.Label(surface, text="Auto uses up to 8 workers. More workers need more RAM.", style="Subtitle.TLabel").grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(surface, text="Boundary patches fill bounded holes from their surrounding surface; input points stay.\nPasses and angle/layer tolerances apply to Plane/Quadratic only. Sizes use input units.\nSet maximum distance beyond the hole radius. Real openings may also be closed.", style="Subtitle.TLabel").grid(row=9, column=0, columnspan=2, sticky="w", pady=(13, 0))

        rebuild = self.rebuild_frame = ttk.LabelFrame(main, text="Build a new surface cloud on a uniform 3D grid", padding=12)
        rebuild.grid(row=2, column=0, rowspan=2, sticky="new", pady=(0, 12))
        rebuild.columnconfigure(1, weight=1)
        ttk.Label(rebuild, text="Grid spacing").grid(row=0, column=0, sticky="w", padx=(0, 16))
        row = ttk.Frame(rebuild)
        row.grid(row=0, column=1, sticky="ew")
        self.rebuild_spacing_entry = self._register(ttk.Entry(row, textvariable=self.rebuild_spacing, width=14))
        self.rebuild_spacing_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="Automatic", variable=self.rebuild_auto, command=self._sync_controls)).pack(side="left", padx=8)
        self._register(ttk.Button(row, text="Estimate spacing", command=self.estimate)).pack(side="right")
        for index, label, variable, values in [
            (1, "Output positions", self.rebuild_positions, ["Exact grid centres (regular 3D lattice)", "Surface-projected (one point per cell)"]),
            (2, "Gap completion", self.rebuild_completion, ["Off", "Balanced", "Strong"])]:
            ttk.Label(rebuild, text=label).grid(row=index, column=0, sticky="w", pady=(10, 0))
            box = self._register(ttk.Combobox(rebuild, textvariable=variable, values=values, state="readonly"))
            box.grid(row=index, column=1, sticky="ew", pady=(10, 0))
            setattr(self, "rebuild_positions_box" if index == 1 else "rebuild_completion_box", box)
        ttk.Label(rebuild, text="Fill reach (input units)").grid(row=3, column=0, sticky="w", pady=(10, 0))
        row = ttk.Frame(rebuild)
        row.grid(row=3, column=1, sticky="ew", pady=(10, 0))
        self.rebuild_reach_entry = self._register(ttk.Entry(row, textvariable=self.rebuild_reach, width=14))
        self.rebuild_reach_entry.pack(side="left")
        self._register(ttk.Checkbutton(row, text="6 x input spacing", variable=self.rebuild_reach_auto, command=self._sync_controls)).pack(side="left", padx=8)
        ttk.Label(rebuild, text="CPU workers").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.rebuild_workers_box = self._register(ttk.Combobox(rebuild, textvariable=self.rebuild_workers,
            values=["Auto", "1", "2", "4", "8", "16", "32"], state="readonly", width=12))
        self.rebuild_workers_box.grid(row=4, column=1, sticky="w", pady=(10, 0))
        ttk.Label(rebuild, text="Maximum output points").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self._register(ttk.Entry(rebuild, textvariable=self.rebuild_limit, width=14)).grid(row=5, column=1, sticky="w", pady=(10, 0))
        ttk.Label(rebuild, text="Creates new sampling instead of retaining irregular input points.\nStrong adds whole-boundary patches; Balanced uses conservative local fits.\nSet reach beyond the hole radius. Real openings may also be closed.\nSurface-projected output follows the surface; exact centres give a true 3D lattice.", style="Subtitle.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 0))

        progress = ttk.Frame(main)
        progress.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        progress.columnconfigure(0, weight=1)
        self.bar = ttk.Progressbar(progress, maximum=100)
        self.bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.log = scrolledtext.ScrolledText(main, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=5, column=0, sticky="nsew")
        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        self.open_button = ttk.Button(actions, text="Open output folder", command=self.open_output, state="disabled")
        self.open_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel_job, state="disabled")
        self.cancel_button.pack(side="right", padx=(8, 0))
        self.run_button = self._register(ttk.Button(actions, text="Regrid and save", command=self.run))
        self.run_button.pack(side="right")
        self._sync_controls()

    def _sync_controls(self,event=None):
        self.size_entry.configure(state="disabled" if self.busy or self.auto.get() else "normal")
        self.voxel_entry.configure(state="disabled" if self.busy or self.voxel_matches_cell.get() else "normal")
        self.delta_entry.configure(state="disabled" if self.busy or not self.keep_below.get() else "normal")
        self.surface_spacing_entry.configure(state="disabled" if self.busy or self.surface_auto.get() else "normal")
        self.surface_gap_entry.configure(state="disabled" if self.busy or self.surface_gap_auto.get() else "normal")
        self.rebuild_spacing_entry.configure(state="disabled" if self.busy or self.rebuild_auto.get() else "normal")
        self.rebuild_reach_entry.configure(state="disabled" if self.busy or self.rebuild_reach_auto.get() else "normal")
        for control in self.surface_local_controls:
            control.configure(state="disabled" if self.busy or self.surface_fit.get().startswith("Boundary") else "normal")
        self.rebuild_frame.grid_remove()
        if self.mode.get() == "3D surface gap filling":
            self.subtitle.set("Fill point-cloud gaps on walls, ground and cave surfaces")
            self.heightfield_frame.grid_remove()
            self.detail_frame.grid_remove()
            self.surface_frame.grid()
            self.run_button.configure(text="Fill 3D gaps and save")
        elif self.mode.get() == "Rebuild uniform 3D cloud":
            self.subtitle.set("New uniform surface sampling with wider gap reconstruction")
            self.heightfield_frame.grid_remove()
            self.detail_frame.grid_remove()
            self.surface_frame.grid_remove()
            self.rebuild_frame.grid()
            self.run_button.configure(text="Rebuild new cloud and save")
        else:
            self.subtitle.set("Heightfield terrain with 3D wall and cave detail")
            self.surface_frame.grid_remove()
            self.heightfield_frame.grid()
            self.detail_frame.grid()
            self.run_button.configure(text="Regrid and save")

    def _mode_changed(self, event=None):
        self._sync_controls()
        source = Path(self.input_path.get().strip())
        if source.is_file():
            suffix = ".las" if source.suffix.lower() == ".txt" else source.suffix
            tag = self._output_tag()
            self.output_path.set(str(source.with_name(source.stem+tag+suffix)))

    def _set_busy(self, busy):
        self.busy = busy
        for widget in self.controls:
            widget.configure(state="disabled" if busy else "normal")
        if not busy:
            self.height_box.configure(state="readonly")
            self.mode_box.configure(state="readonly")
            self.surface_fit_box.configure(state="readonly")
            self.surface_workers_box.configure(state="readonly")
            self.rebuild_positions_box.configure(state="readonly")
            self.rebuild_completion_box.configure(state="readonly")
            self.rebuild_workers_box.configure(state="readonly")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.open_button.configure(state="normal" if self.output_directory and not busy else "disabled")
        self._sync_controls()

    def browse_input(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open point cloud", filetypes=FILE_TYPES)
        if path:
            self.input_path.set(path)
            source = Path(path)
            suffix = ".las" if source.suffix.lower() == ".txt" else source.suffix
            tag = self._output_tag()
            self.output_path.set(str(source.with_name(source.stem + tag + suffix)))

    def _output_tag(self):
        return {"3D surface gap filling": "_surface_filled", "Rebuild uniform 3D cloud": "_uniform"}.get(self.mode.get(), "_hybrid")

    def browse_output(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="Save regridded point cloud", filetypes=FILE_TYPES, defaultextension=".las", initialfile=Path(self.output_path.get()).name or "regridded.las", confirmoverwrite=False)
        if path:
            self.output_path.set(path)

    def _input(self):
        path = Path(self.input_path.get().strip()).expanduser()
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            raise ValueError("Choose an existing LAS, LAZ, PLY, XYZ or TXT input file.")
        return path.resolve()

    def _launch(self, action):
        self.cancel.clear()
        self.bar["value"] = 0
        self._set_busy(True)
        def worker():
            try:
                value = action()
                self.events.put(("done", value))
            except Cancelled:
                self.events.put(("cancelled", None))
            except Exception as error:
                self.events.put(("error", (str(error), traceback.format_exc())))
        Thread(target=worker, daemon=True, name="hybrid-regrid").start()

    def estimate(self):
        if self.busy:
            return
        try:
            path = self._input()
        except ValueError as error:
            messagebox.showerror("Input", str(error), parent=self.root)
            return
        selected_mode = self.mode.get()
        worker_setting = self.rebuild_workers.get() if selected_mode == "Rebuild uniform 3D cloud" else self.surface_workers.get()
        query_workers = resolve_workers(0 if worker_setting == "Auto" else int(worker_setting)) if selected_mode != "Hybrid Regrid" else -1
        def job():
            self.events.put(("progress", (0, "Reading cloud to estimate spacing")))
            cloud = read_cloud(path, self.cancel, lambda message: self.events.put(("log", message)))
            self.events.put(("progress", (0.5, "Estimating nearest-neighbour spacing")))
            estimate = estimate_native_spacing if selected_mode == "Rebuild uniform 3D cloud" else auto_voxel_size
            size = estimate(cloud.xyz, self.cancel, workers=query_workers)
            return {"estimate": size, "points": len(cloud.xyz), "estimate_mode": selected_mode}
        self._launch(job)

    def run(self):
        if self.busy:
            return
        try:
            source = self._input()
            if not self.output_path.get().strip():
                raise ValueError("Choose an output path.")
            output = Path(self.output_path.get().strip()).expanduser().resolve()
            if output == source or (output.exists() and os.path.samefile(output, source)):
                raise ValueError("Choose an output different from the input; the scan is kept intact.")
            if output.suffix.lower() not in SUPPORTED or not output.parent.is_dir():
                raise ValueError("Choose an existing output folder and a .las, .laz, .ply, .xyz or .txt extension.")
            surface_mode = self.mode.get() == "3D surface gap filling"
            rebuild_mode = self.mode.get() == "Rebuild uniform 3D cloud"
            if rebuild_mode:
                params = RebuildParameters(
                    spacing=None if self.rebuild_auto.get() else float(self.rebuild_spacing.get()),
                    positions="grid" if self.rebuild_positions.get().startswith("Exact") else "surface",
                    completion=self.rebuild_completion.get().lower(),
                    reach=None if self.rebuild_reach_auto.get() else float(self.rebuild_reach.get()),
                    workers=0 if self.rebuild_workers.get() == "Auto" else int(self.rebuild_workers.get()),
                    max_output_points=int(self.rebuild_limit.get()))
            elif surface_mode:
                params = SurfaceParameters(
                    spacing=None if self.surface_auto.get() else float(self.surface_spacing.get()),
                    max_gap_distance=None if self.surface_gap_auto.get() else float(self.surface_gap.get()),
                    passes=int(self.surface_passes.get()), max_angle_deg=float(self.surface_angle.get()),
                    layer_tolerance=float(self.surface_layer.get()),
                    interpolation="plane" if self.surface_fit.get().startswith("Plane") else "quadratic",
                    max_added_points=int(self.surface_max_added.get()),
                    workers=0 if self.surface_workers.get() == "Auto" else int(self.surface_workers.get()),
                    method="boundary" if self.surface_fit.get().startswith("Boundary") else "local")
            else:
                params = Parameters(
                    cell_size=None if self.auto.get() else float(self.cell_size.get()),
                    voxel_size=None if self.voxel_matches_cell.get() else float(self.voxel_size.get()),
                    height_stat=self.height_stat.get().split()[0].lower(),
                    fill_passes=int(self.iterations.get()), wall_slope_deg=float(self.wall_slope.get()),
                    keep_below_surface=self.keep_below.get(), below_delta_mul=float(self.below_delta.get()),
                    max_added_points=int(self.max_added.get()))
            params.validate()
        except (ValueError, OSError) as error:
            messagebox.showerror("Check settings", str(error), parent=self.root)
            return
        exists = output.exists() or Path(str(output) + ".regrid.json").exists()
        if exists and not messagebox.askyesno("Replace output?", f"Replace the existing output and/or report?\n\n{output}", parent=self.root):
            return
        self._append_log(f"\nProcessing {source.name}")
        last_update = [0.0]
        def progress(fraction, message):
            now = time.monotonic()
            if fraction >= 1 or now - last_update[0] >= 0.08:
                last_update[0] = now
                self.events.put(("progress", (fraction, message)))
        def job():
            process = process_rebuild_file if rebuild_mode else process_surface_file if surface_mode else process_file
            return process(source, output, params, self.cancel, progress, lambda message: self.events.put(("log", message)), overwrite=exists)
        self._launch(job)

    def cancel_job(self):
        self.cancel.set()
        self.status.set("Cancelling after the current processing step…")
        self.cancel_button.configure(state="disabled")

    def _append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            for _ in range(150):
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.bar["value"] = value[0] * 100
                    if not self.cancel.is_set():
                        self.status.set(value[1])
                elif kind == "log":
                    self._append_log(value)
                else:
                    self._set_busy(False)
                    if kind == "done":
                        self.bar["value"] = 100
                        if "estimate" in value:
                            surface = value.get("estimate_mode") == "3D surface gap filling"
                            uniform = value.get("estimate_mode") == "Rebuild uniform 3D cloud"
                            variable = self.rebuild_spacing if uniform else self.surface_spacing if surface else self.cell_size
                            variable.set(f"{value['estimate']:.10g}")
                            self.status.set(f"Estimated spacing: {value['estimate']:.6g} input units")
                            self._append_log(f"Spacing estimate from {value['points']:,} points: {value['estimate']:.10g}")
                            if surface:
                                self.surface_gap.set(f"{3*value['estimate']:.10g}")
                            if uniform:
                                self.rebuild_reach.set(f"{6*value['estimate']:.10g}")
                        else:
                            self.output_directory = Path(value["output"]).parent
                            self.open_button.configure(state="normal")
                            self.status.set(f"Saved {value['output_points']:,} points, including {value['added_points']:,} added estimates.")
                            if value.get("mode") == "uniform_rebuild_3d":
                                self.rebuild_spacing.set(f"{value['resolved_spacing']:.10g}")
                                self.rebuild_reach.set(f"{value['resolved_reach']:.10g}")
                            elif value.get("mode") == "surface_fill_3d":
                                self.surface_spacing.set(f"{value['resolved_spacing']:.10g}")
                                self.surface_gap.set(f"{value['resolved_max_gap_distance']:.10g}")
                            else:
                                self.cell_size.set(f"{value['resolved_cell_size']:.10g}")
                                self.voxel_size.set(f"{value['resolved_voxel_size']:.10g}")
                    elif kind == "cancelled":
                        self.status.set("Cancelled. Output file was not replaced.")
                        self._append_log("Cancelled before committing output.")
                    elif kind == "error":
                        self.status.set("Processing failed; see log.")
                        self._append_log(value[1])
                        if not self.closing:
                            messagebox.showerror("Processing failed", value[0], parent=self.root)
                    if self.closing:
                        self.root.destroy()
                        return
        except Empty:
            pass
        self.root.after(80, self._poll)

    def open_output(self):
        if self.output_directory:
            if sys.platform == "win32":
                os.startfile(self.output_directory)
            else:
                import webbrowser
                webbrowser.open(self.output_directory.as_uri())

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel_job()
        else:
            self.root.destroy()

    def about(self):
        messagebox.showinfo("About Hybrid Regrid", "Hybrid Regrid Tk — test version 0.6\nCopyright (c) 2025–2026 Przemysław Mielniczenko\nMIT License\n\n"
            "Boundary patches complete bounded holes using connected surface charts and a curved trend with a smooth boundary residual. Available in Strong rebuild and the surface mode's Boundary patches method. Genuine openings can also be closed.\n\n"
            "Rebuild uniform 3D cloud creates new sampling on a 3D surface grid, with Off/Balanced/Strong completion and wider directional support. Choose exact voxel centres or surface-projected points. Input files are kept intact.\n\n"
            "3D surface gap filling keeps input points and adds estimates using locally oriented plane/quadratic fits. Measured samples must surround each candidate and pass normal/depth checks.\n\n"
            "Ports the editor's Hybrid Regrid: robust Z per XY column, small 2D hole filling, and 3D voxel samples retained for steep cells or below the heightfield. Z is the up axis.\n\n"
            "The wall rule uses height differences, not point normals. Isolated walls and geometry above the selected height can be omitted. Heightfield and voxel samples may overlap.\n\n"
            "Works in memory. Start with a cropped scan. Cancellation is checked between processing steps. Input files are kept intact.\n\n"
            "LAS/LAZ retains CRS and representative source fields. PLY keeps supported vertex attributes; faces and normals are omitted. XYZ keeps XYZ/RGB only.\n\n"
            "Instructions: START-HERE.txt\nLicenses: LICENSE and licenses/", parent=self.root)



def main():
    root = tk.Tk()
    RegridApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
