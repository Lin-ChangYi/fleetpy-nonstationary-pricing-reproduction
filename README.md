# Nonstationary pricing reproduction code

This release reproduces the reported Daily-10,000 nonstationary pricing experiment:

- six adaptive methods: Beta-TS, EXP3, and Hedge, each with HT-IX or simple-average feedback;
- explore-then-commit (ETC) benchmarks with 5, 10, 14, and 24 observed feedback updates;
- 50 paired daily simulation samples with identical demand traces, fleet seeds, and MAB seeds across methods within each sample.

The release includes a patch for the FleetPy runtime, the experimental pricing controller and learners, the passenger-acceptance model, the custom 42-cell grid, and the scripts to prepare, run, and analyze the batches.

## Data and framework

The simulations use the public FleetPy Manhattan data package, version 1.0, published on 2025-04-10: [Zenodo record 15187906](https://zenodo.org/records/15187906), DOI [10.5281/zenodo.15187906](https://doi.org/10.5281/zenodo.15187906). The archive is `FleetPy_Manhattan.zip` with MD5 `8b11882ae9c6d87f666bf6e006806744`.

The experiment uses the `2018-11-12.csv` full-day matched-demand file. The FleetPy Manhattan package derives these inputs from NYC Taxi and Limousine Commission trip records for 2018-11-11 through 2018-11-18 after retaining trips whose origin and destination are both in Manhattan. The road network is extracted from OpenStreetMap; FleetPy supplies preprocessed travel-time and distance matrices. The full preprocessing specification is in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

The experiment uses [FleetPy 1.0.1 at commit `e4a5bae75eadd58abb67b909f7e2c76f4830b33c`](https://github.com/TUM-VT/FleetPy/commit/e4a5bae75eadd58abb67b909f7e2c76f4830b33c) with the [research runtime patch](fleetpy_base/research.patch). The patch carries the experiment's request-level pricing and acceptance feedback, deterministic acceptance draw, memory-mapped network matrices, and single-occupancy vehicle definition. A working Gurobi installation and license are required by the Alonso--Mora assignment optimizer.

## Installation

Run the following commands from the root of this repository in PowerShell. The setup requires Git and Conda.

1. Create and activate the environment.

```powershell
conda env create -f environment.yml
conda activate fleetpy-nonstationary-pricing
```

2. Prepare the experiment's FleetPy checkout. Pick a destination that does not yet exist. The script downloads the pinned commit from the official public repository, checks the patch, and applies it to the checkout.

```powershell
$root = "C:\path\to\FleetPy-experiment"
python .\tools\prepare_fleetpy_base.py --fleetpy-root $root
```

3. Extract the public `FleetPy_Manhattan.zip`. Copy the contents of its `FleetPy_Manhattan` folder into the checkout's `data` directory.

```powershell
$manhattan = "C:\path\to\extracted\FleetPy_Manhattan"
Copy-Item -Path "$manhattan\*" -Destination "$root\data" -Recurse -Force
```

The paths below are relative to `$root`.

```text
data/demand/Manhattan_2018/matched/Manhattan_2019_corrected/2018-11-12.csv
data/networks/Manhattan_2019_corrected/base/nodes.csv
data/networks/Manhattan_2019_corrected/ff/tables/nn_fastest_tt.npy
data/networks/Manhattan_2019_corrected/ff/tables/nn_fastest_distance.npy
data/networks/Manhattan_2019_corrected/2018-11-12_mean_tt_factor_hourly.csv
data/zones/Manhattan_Taxi_Zones/Manhattan_2019_corrected/node_zone_info.csv
```

4. Run either batch. The script installs the overlay, verifies the taxi-zone mapping, prepares its centroids, verifies the paired demand and forecast inputs, and writes the scenario CSV. Add `-Execute` to start the simulation.

```powershell
.\run\run_experiment.ps1 -Batch bandits -FleetPyRoot $root -Samples 1
.\run\run_experiment.ps1 -Batch bandits -FleetPyRoot $root -Samples 1 -Execute
.\run\run_experiment.ps1 -Batch etc -FleetPyRoot $root -Samples 1 -Execute
```

`-Samples 1` uses a single full-day sample for all six bandit methods or all four ETC benchmarks. Use `-Workers 1` to run serially. Omit `-Samples` for the published 50-sample design.

## Outputs

The runner writes restartable checkpoints under `experiments/checkpoints/` and FleetPy results under `studies/nonstationary_pricing_reproduction/results/` in the target FleetPy checkout. A completed scenario must include a pricing request log with the columns used by the analysis and at least one data row.

Pass the FleetPy checkout to locate the checkpoints automatically.

```powershell
$root = "C:\path\to\FleetPy-experiment"
python .\analysis\analyze_daily10000_logit_switch_f400_p3_bandits.py `
  --fleetpy-root $root `
  --output-dir "$root\analysis\outputs\nonstationary_bandits"

python .\analysis\analyze_daily10000_logit_switch_f400_p3_bandits_vs_traditional_etc.py `
  --fleetpy-root $root `
  --output-dir "$root\analysis\outputs\nonstationary_bandits_vs_etc"
```

You can also pass full checkpoint paths. An explicit path takes precedence over the path derived from `--fleetpy-root` for that checkpoint.

```powershell
python .\analysis\analyze_daily10000_logit_switch_f400_p3_bandits.py `
  --checkpoint "$root\experiments\checkpoints\NONSTATIONARY_P3_R2_BANDITS_50x6.csv" `
  --output-dir "$root\analysis\outputs\nonstationary_bandits"

python .\analysis\analyze_daily10000_logit_switch_f400_p3_bandits_vs_traditional_etc.py `
  --bandit-checkpoint "$root\experiments\checkpoints\NONSTATIONARY_P3_R2_BANDITS_50x6.csv" `
  --traditional-checkpoint "$root\experiments\checkpoints\NONSTATIONARY_P3_R2_ETC_50x4.csv" `
  --output-dir "$root\analysis\outputs\nonstationary_bandits_vs_etc"
```

## Repository layout

```text
analysis/       paired realized-revenue summaries and figures
config/         five-phase simulation configuration and frozen 50-sample manifest
data_assets/    custom 1,500 m pricing grid and node-to-grid mapping
fleetpy_base/   research runtime patch for the pinned public FleetPy version
overlay/        FleetPy pricing controller, acceptance model, and MAB algorithms
run/            restartable batch runner and PowerShell entry point
tools/          base preparation, overlay installation, input and scenario preparation
```

The public data are not redistributed here. The recorded hashes, deterministic selection rule, and public source make the generated inputs auditable without including a duplicate copy of the demand data.
