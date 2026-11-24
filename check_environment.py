"""Check the pinned versions, rather than merely checking that imports work."""
from importlib import import_module, metadata
from pathlib import Path
import sys


def check(requirements: Path | None = None) -> list[str]:
    requirements = requirements or Path(__file__).with_name("requirements.txt")
    problems = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==", 1)
        package = package.split("[", 1)[0]
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package} is missing (requires {expected})")
        else:
            if actual != expected:
                problems.append(f"{package}: installed {actual}, requires {expected}")
    for name in ("numpy", "scipy.spatial", "laspy", "lazrs", "threadpoolctl", "tkinter"):
        try:
            import_module(name)
        except (ImportError, OSError) as error:
            problems.append(f"Cannot import {name}: {error}")
    if not problems:
        import laspy
        if not laspy.LazBackend.detect_available():
            problems.append("No usable LAZ compression backend was detected.")
    return problems


if __name__ == "__main__":
    issues = check()
    for issue in issues:
        print(issue)
    if not issues:
        print("Pinned dependencies and LAZ backend verified.")
    sys.exit(1 if issues else 0)
