import argparse
import ast
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_tree(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def register_module(source, function_name, registry, class_name, module):
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    function = next((node for node in functions if node.name == function_name), None)
    if function is None or not isinstance(function.body[-1], ast.Return):
        raise ValueError(f"Cannot locate the FleetPy registry function {function_name}.")

    lines = source.splitlines(keepends=True)
    return_line = function.body[-1].lineno - 1
    indentation = lines[return_line][:function.body[-1].col_offset]
    entry = f'{indentation}{registry}["{class_name}"] = ("{module}", "{class_name}")\n'
    for statement in function.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == registry
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == class_name
            for target in statement.targets
        ):
            lines[statement.lineno - 1:statement.end_lineno] = [entry]
            break
    else:
        lines.insert(return_line, entry)
    return "".join(lines)


def install(fleetpy_root):
    fleetpy_root = fleetpy_root.resolve()
    registry_path = fleetpy_root / "src" / "misc" / "init_modules.py"
    traveler_path = fleetpy_root / "src" / "demand" / "TravelerModels.py"
    fleet_control_path = fleetpy_root / "src" / "fleetctrl" / "FleetControlBase.py"
    required = [
        registry_path,
        traveler_path,
        fleet_control_path,
        fleetpy_root / "src" / "fleetctrl" / "pricing",
        fleetpy_root / "data" / "zones",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Not a FleetPy checkout:\n" + "\n".join(missing))

    if (
        "def _stable_uniform_01(" not in traveler_path.read_text(encoding="utf-8")
        or ".record_request_acceptance(" not in fleet_control_path.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "FleetPy is missing the research runtime. Prepare a new checkout with "
            "tools/prepare_fleetpy_base.py before running this experiment."
        )
    registry = registry_path.read_text(encoding="utf-8")
    registry = register_module(
        registry, "get_src_request_modules", "rm_dict",
        "TimeVaryingLogitPriceSensitiveIndividualConstraintRequest", "src.demand.TimeVaryingAcceptance",
    )
    registry = register_module(
        registry, "get_src_dynamic_pricing_strategies", "dp_dict",
        "MABSpatialPricing", "src.fleetctrl.pricing.MABSpatialPricing",
    )
    ast.parse(registry, filename=str(registry_path))

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
    registry_path.write_text(registry, encoding="utf-8")
    print(f"Installed the nonstationary-pricing overlay in {fleetpy_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        install(args.fleetpy_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not install the overlay: {exc}\n")


if __name__ == "__main__":
    main()
