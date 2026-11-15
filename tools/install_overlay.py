import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_tree(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def install_request_registry(path):
    text = path.read_text(encoding="utf-8")
    line = (
        '    rm_dict["TimeVaryingLogitPriceSensitiveIndividualConstraintRequest"] '
        '= ("src.demand.TimeVaryingAcceptance", '
        '"TimeVaryingLogitPriceSensitiveIndividualConstraintRequest")'
    )
    pattern = r'^\s*rm_dict\["TimeVaryingLogitPriceSensitiveIndividualConstraintRequest"\].*$'
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, line, text, flags=re.MULTILINE)
    else:
        anchor = '    rm_dict["ProbabilisticPriceSensitiveIndividualConstraintRequest"] = ("src.demand.TravelerModels", "ProbabilisticPriceSensitiveIndividualConstraintRequest")'
        if anchor not in text:
            raise ValueError(f"Cannot locate request-module registry entry in {path}")
        text = text.replace(anchor, f"{anchor}\n{line}")
    path.write_text(text, encoding="utf-8")


def install_pricing_registry(path):
    text = path.read_text(encoding="utf-8")
    line = '    dp_dict["MABSpatialPricing"] = ("src.fleetctrl.pricing.MABSpatialPricing", "MABSpatialPricing")'
    pattern = r'^\s*dp_dict\["MABSpatialPricing"\].*$'
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, line, text, flags=re.MULTILINE)
    else:
        anchor = '    dp_dict["RandomizedPricing"] = ('
        if anchor not in text:
            raise ValueError(f"Cannot locate pricing-module registry entry in {path}")
        text = text.replace(anchor, f"{line}\n{anchor}")
    path.write_text(text, encoding="utf-8")


def install(fleetpy_root):
    fleetpy_root = fleetpy_root.resolve()
    required = [
        fleetpy_root / "src" / "misc" / "init_modules.py",
        fleetpy_root / "src" / "fleetctrl" / "pricing",
        fleetpy_root / "src" / "demand",
        fleetpy_root / "data" / "zones",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Not a FleetPy checkout:\n" + "\n".join(missing))

    overlay = ROOT / "overlay"
    shutil.copy2(
        overlay / "src" / "fleetctrl" / "pricing" / "MABSpatialPricing.py",
        fleetpy_root / "src" / "fleetctrl" / "pricing" / "MABSpatialPricing.py",
    )
    shutil.copy2(
        overlay / "src" / "demand" / "TimeVaryingAcceptance.py",
        fleetpy_root / "src" / "demand" / "TimeVaryingAcceptance.py",
    )
    copy_tree(overlay / "mab_algo", fleetpy_root / "mab_algo")
    copy_tree(
        ROOT / "data_assets" / "Manhattan_Grid_1500m",
        fleetpy_root / "data" / "zones" / "Manhattan_Grid_1500m",
    )
    scenario_dir = fleetpy_root / "studies" / "nonstationary_pricing_reproduction" / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config" / "nonstationary_p3_r2.yaml", scenario_dir / "const_cfg_nonstationary_p3_r2.yaml")
    install_request_registry(fleetpy_root / "src" / "misc" / "init_modules.py")
    install_pricing_registry(fleetpy_root / "src" / "misc" / "init_modules.py")
    print(f"Installed the nonstationary-pricing overlay in {fleetpy_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.fleetpy_root)


if __name__ == "__main__":
    main()
