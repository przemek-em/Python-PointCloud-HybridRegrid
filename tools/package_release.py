"""Create a clean GitHub source archive using only the project's allowlist.

Requires the Python standard library only. Every run uses a fresh staging
folder; existing staging folders and archives are preserved.
"""
from hashlib import sha256
from pathlib import Path
import ast
import json
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.6"
PACKAGE = f"HybridRegridTk-{VERSION}"
ZIP_TIME = (2026, 11, 24, 0, 0, 0)
ROOT_FILES = (
    "README.md", "LICENSE", "START-HERE.txt", ".gitignore", ".gitattributes",
    "requirements.txt", "run.bat", "app.py", "check_environment.py", "cloud.py",
    "cloud_io.py", "ply_io.py", "cpu_workers.py", "hybrid.py", "jobs.py", "surface_fill.py",
    "uniform_rebuild.py", "boundary_patch.py", ".github/workflows/tests.yml",
    "tools/package_release.py", "assets/boundary-patch-demo.png",
    "examples/EXAMPLES.txt", "examples/large_planar_gap.ply",
    "examples/vertical_wall_gap.ply", "examples/curved_cave_gap.ply",
)


def collect_files():
    files = [ROOT / name for name in ROOT_FILES]
    files.extend(sorted((ROOT / "tests").glob("test_*.py")))
    files.append(ROOT / "tests/ui_smoke.py")
    files.extend(sorted(path for path in (ROOT / "licenses").rglob("*") if path.is_file()))
    payloads = {}
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing file or unsupported symbolic link: {path}")
        path.resolve().relative_to(ROOT)
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".las", ".laz", ".vdb", ".nvdb", ".log"}:
            raise ValueError(f"Data/debug file is not distributable: {relative}")
        if path.relative_to(ROOT).parts[0] in {"diagnostics", "dist", ".venv", ".git"}:
            raise ValueError(f"Local-only directory is not distributable: {relative}")
        if relative in payloads:
            raise ValueError(f"Duplicate package path: {relative}")
        payloads[relative] = path.read_bytes()
    for name, data in payloads.items():
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(data.decode("utf-8"), filename=name)):
            modules = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                local = module.split(".")[0]+".py"
                if (ROOT/local).is_file() and local not in payloads:
                    raise ValueError(f"{name} imports local module {local}, missing from the allowlist.")
    return dict(sorted(payloads.items()))


def main():
    payloads = collect_files()
    manifest = {
        "project": "Hybrid Regrid Tk",
        "version": VERSION,
        "distribution": "GitHub source",
        "python": "64-bit Python 3.11 or 3.12",
        "dependencies": dict(line.split("==", 1) for line in payloads["requirements.txt"].decode("utf-8").splitlines() if line.strip()),
        "files": [
            {"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in payloads.items()
        ],
    }
    payloads["PACKAGE-MANIFEST.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False)+"\n").encode("utf-8")

    distribution = ROOT / "dist"
    revision = 1
    while True:
        suffix = "" if revision == 1 else f"-r{revision}"
        stage = distribution / "github" / (PACKAGE+suffix)
        archive_path = distribution / f"{PACKAGE}{suffix}-github.zip"
        if not stage.exists() and not archive_path.exists():
            break
        revision += 1
    stage.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            entry = zipfile.ZipInfo(f"{PACKAGE}/{name}", ZIP_TIME)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity verification failed.")
        expected = {f"{PACKAGE}/{name}" for name in payloads}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Archive file list does not match the allowlist.")
        for item in manifest["files"]:
            archived = archive.read(f"{PACKAGE}/{item['path']}")
            staged = (stage / item["path"]).read_bytes()
            if len(archived) != item["bytes"] or sha256(archived).hexdigest() != item["sha256"] or staged != archived:
                raise ValueError(f"File hash verification failed: {item['path']}")

    digest = sha256(archive_path.read_bytes()).hexdigest()
    checksum = archive_path.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    print(json.dumps({
        "archive": str(archive_path), "source_folder": str(stage),
        "checksum_file": str(checksum), "sha256": digest,
        "files": len(payloads), "bytes": archive_path.stat().st_size,
        "verification": "Archive CRC, file allowlist, staged bytes and SHA-256 hashes passed",
    }, indent=2))


if __name__ == "__main__":
    main()
