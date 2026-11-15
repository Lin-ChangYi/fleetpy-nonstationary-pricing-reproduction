# Nonstationary pricing reproduction code

This release reproduces the reported Daily-10,000 nonstationary pricing experiment:

- six adaptive methods: Beta-TS, EXP3, and Hedge, each with HT-IX or simple-average feedback;
- explore-then-commit (ETC) benchmarks with 5, 10, 14, and 24 observed feedback updates;
- 50 paired daily simulation samples with identical demand traces, fleet seeds, and MAB seeds across methods within each sample.

The release is an overlay for FleetPy rather than a redistributed copy of FleetPy or the public Manhattan data. It includes the research-specific controller, learner implementations, passenger-acceptance model, scenario builders, custom 42-cell grid, runner, and analysis scripts.

## Data and framework

The simulations use the public FleetPy Manhattan data package, version 1.0, published on 2025-04-10: [Zenodo record 15187906](https://zenodo.org/records/15187906), DOI [10.5281/zenodo.15187906](https://doi.org/10.5281/zenodo.15187906). The archive is `FleetPy_Manhattan.zip` with MD5 `8b11882ae9c6d87f666bf6e006806744`.

The experiment uses the `2018-11-12.csv` full-day matched-demand file. The FleetPy Manhattan package derives these inputs from NYC Taxi and Limousine Commission trip records for 2018-11-11 through 2018-11-18 after retaining trips whose origin and destination are both in Manhattan. The road network is extracted from OpenStreetMap; FleetPy supplies preprocessed travel-time and distance matrices. The full preprocessing specification is in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

This overlay was verified against the FleetPy research checkout at commit `b23378b5bfb22d38742c583a99a19b4772373227`. A working Gurobi installation and license are required by the Alonso--Mora assignment optimizer.

## Installation

1. Create the environment:

```powershell
conda env create -f environment.yml
conda activate fleetpy-nonstationary-pricing
```

2. Obtain a FleetPy checkout and install the public Manhattan data package into its `data/` directory. The following files must be present:

```text
data/demand/Manhattan_2018/matched/Manhattan_2019_corrected/2018-11-12.csv
data/networks/Manhattan_2019_corrected/base/nodes.csv
data/networks/Manhattan_2019_corrected/ff/tables/nn_fastest_tt.npy
data/networks/Manhattan_2019_corrected/2018-11-12_mean_tt_factor_hourly.csv
data/zones/Manhattan_Taxi_Zones/Manhattan_2019_corrected/node_zone_info.csv
```

3. Run either experiment from this release directory. The command installs the overlay, recreates the paired demand and forecast inputs, writes the scenario CSV, then prepares or runs the batch.

```powershell
.\run\run_experiment.ps1 -Batch bandits -FleetPyRoot C:\path\to\FleetPy
.\run\run_experiment.ps1 -Batch bandits -FleetPyRoot C:\path\to\FleetPy -Execute
.\run\run_experiment.ps1 -Batch etc -FleetPyRoot C:\path\to\FleetPy -Execute
```

Use `-Samples 1` for a smoke test or `-Workers 1` to run serially. The default `-Samples 50` reconstructs the published paired design. The script validates SHA-256 hashes of the regenerated inputs against the recorded study inputs.

## Outputs

The runner writes restartable checkpoints under `experiments/checkpoints/` and FleetPy results under `studies/nonstationary_pricing_reproduction/results/` in the target FleetPy checkout. Analyze completed batches with explicit paths:

```powershell
$root = "C:\path\to\FleetPy"
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
overlay/        FleetPy pricing controller, acceptance model, and MAB algorithms
run/            restartable batch runner and PowerShell entry point
tools/          overlay installer, demand/forecast reconstruction, scenario builder
```

The public data are not redistributed here. The recorded hashes, deterministic selection rule, and public source make the generated inputs auditable without including a duplicate copy of the demand data.
