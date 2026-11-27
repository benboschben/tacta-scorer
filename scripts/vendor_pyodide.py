"""Copy the parts of a Pyodide release that the web app needs into a folder,
so the site serves Python and OpenCV itself instead of loading them from a CDN.

    python3 scripts/vendor_pyodide.py               # into web/pyodide/ for local use
    python3 scripts/vendor_pyodide.py --out _site/pyodide

The release (about 340 MB) is downloaded once into .cache/pyodide/; only the
runtime and the wheels in PACKAGES, with their dependencies, are copied out.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "314.0.7"
URL = f"https://github.com/pyodide/pyodide/releases/download/{VERSION}/pyodide-{VERSION}.tar.bz2"
# The runtime: the ES module loader (Pyodide 314 runs only in module workers)
# and what loadPyodide() fetches before any package.
CORE = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
# What tacta_score.py imports, by Pyodide package name.
PACKAGES = ["numpy", "opencv-python"]

ROOT = Path(__file__).resolve().parent.parent


def download(cache):
    tarball = cache / f"pyodide-{VERSION}.tar.bz2"
    if not tarball.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}", file=sys.stderr)
        part = tarball.with_suffix(".part")
        with urllib.request.urlopen(URL) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f)
        part.rename(tarball)
    return tarball


def extract(tarball, names, out):
    """Copy the named files from the release's pyodide/ folder into `out`."""
    wanted = {f"pyodide/{n}": n for n in names}
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:bz2") as tar:
        for member in tar:
            if member.name in wanted and member.isfile():
                with tar.extractfile(member) as src, open(out / wanted.pop(member.name), "wb") as dst:
                    shutil.copyfileobj(src, dst)
    if wanted:
        sys.exit(f"Not in the Pyodide {VERSION} release: {', '.join(wanted.values())}")


def wheels(lock):
    """File name and sha256 of each package in PACKAGES and its dependencies."""
    packages = lock["packages"]
    found, stack = {}, list(PACKAGES)
    while stack:
        name = stack.pop()
        if name not in found:
            found[name] = packages[name]
            stack += packages[name].get("depends", [])
    return {p["file_name"]: p["sha256"] for p in found.values()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "pyodide")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "pyodide")
    args = parser.parse_args()

    tarball = download(args.cache)
    # Two passes over the archive: the lock file says which wheels to take.
    extract(tarball, CORE, args.out)
    needed = wheels(json.loads((args.out / "pyodide-lock.json").read_text()))
    extract(tarball, list(needed), args.out)
    for name, sha in needed.items():
        if hashlib.sha256((args.out / name).read_bytes()).hexdigest() != sha:
            sys.exit(f"{name} doesn't match its checksum in pyodide-lock.json")

    total = sum(f.stat().st_size for f in args.out.iterdir())
    print(f"Pyodide {VERSION} with {', '.join(PACKAGES)} in {args.out} ({total / 1e6:.0f} MB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
