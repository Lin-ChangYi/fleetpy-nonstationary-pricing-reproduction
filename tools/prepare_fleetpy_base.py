import argparse
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PATCH = ROOT / "fleetpy_base" / "research.patch"
UPSTREAM_URL = "https://github.com/TUM-VT/FleetPy.git"
UPSTREAM_VERSION = "1.0.1"
UPSTREAM_COMMIT = "e4a5bae75eadd58abb67b909f7e2c76f4830b33c"


def prepare_fleetpy_base(fleetpy_root):
    fleetpy_root = fleetpy_root.expanduser()
    if fleetpy_root.is_symlink():
        raise FileExistsError(f"Choose a new FleetPy directory. Already exists: {fleetpy_root}")
    fleetpy_root = fleetpy_root.resolve()
    if fleetpy_root.exists():
        raise FileExistsError(f"Choose a new FleetPy directory. Already exists: {fleetpy_root}")
    if not RESEARCH_PATCH.is_file():
        raise FileNotFoundError(f"Missing research patch: {RESEARCH_PATCH}")
    research_patch = RESEARCH_PATCH.read_bytes()

    fleetpy_root.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = str(fleetpy_root.parent)
    if os.name == "nt" and not temp_parent.startswith("\\\\?\\"):
        # Allow removal of deeply nested upstream example files on failure.
        if temp_parent.startswith("\\\\"):
            temp_parent = "\\\\?\\UNC\\" + temp_parent[2:]
        else:
            temp_parent = "\\\\?\\" + temp_parent
    with tempfile.TemporaryDirectory(prefix=".fleetpy-base-", dir=temp_parent) as temp:
        checkout = Path(temp) / "FleetPy"
        checkout.mkdir()
        git = ["git", "-C", str(checkout)]
        subprocess.run([*git, "init", "--quiet"], check=True)
        subprocess.run([*git, "config", "--local", "core.autocrlf", "false"], check=True)
        subprocess.run([*git, "config", "--local", "core.longpaths", "true"], check=True)
        subprocess.run([*git, "remote", "add", "origin", UPSTREAM_URL], check=True)
        subprocess.run(
            [*git, "fetch", "--quiet", "--no-tags", "--depth=1", "origin", UPSTREAM_COMMIT],
            check=True,
        )
        subprocess.run([*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if head != UPSTREAM_COMMIT:
            raise ValueError(f"Expected FleetPy commit {UPSTREAM_COMMIT}, found {head}")

        apply_patch = [*git, "apply", "--whitespace=nowarn"]
        subprocess.run([*apply_patch, "--check", "-"], input=research_patch, check=True)
        subprocess.run([*apply_patch, "-"], input=research_patch, check=True)
        if fleetpy_root.exists() or fleetpy_root.is_symlink():
            raise FileExistsError(f"FleetPy directory appeared during preparation: {fleetpy_root}")
        checkout.rename(fleetpy_root)

    return fleetpy_root


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the pinned FleetPy checkout and apply the experiment's research patch.",
    )
    parser.add_argument("--fleetpy-root", type=Path, required=True, metavar="NEW_DIRECTORY")
    args = parser.parse_args()
    try:
        fleetpy_root = prepare_fleetpy_base(args.fleetpy_root)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        parser.exit(1, f"Could not prepare FleetPy: {exc}\n")
    print(f"Prepared FleetPy {UPSTREAM_VERSION} ({UPSTREAM_COMMIT}) at {fleetpy_root}")


if __name__ == "__main__":
    main()
