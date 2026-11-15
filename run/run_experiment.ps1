param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bandits", "etc")]
    [string]$Batch,
    [Parameter(Mandatory = $true)]
    [string]$FleetPyRoot,
    [string]$Python = "python",
    [ValidateRange(1, 50)]
    [int]$Samples = 50,
    [ValidateRange(1, 12)]
    [int]$Workers = 4,
    [ValidateRange(0.5, 64.0)]
    [double]$MemoryGuardGiB = 2.0,
    [switch]$Execute
)

$releaseRoot = Split-Path -Parent $PSScriptRoot
$FleetPyRoot = (Resolve-Path -LiteralPath $FleetPyRoot).Path

& $Python "$releaseRoot\tools\install_overlay.py" --fleetpy-root $FleetPyRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$releaseRoot\tools\prepare_paired_inputs.py" --fleetpy-root $FleetPyRoot --samples $Samples --verify-hashes
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$releaseRoot\tools\make_scenarios.py" --fleetpy-root $FleetPyRoot --samples $Samples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:FLEETPY_ROOT = $FleetPyRoot
$constantConfig = "$FleetPyRoot\studies\nonstationary_pricing_reproduction\scenarios\const_cfg_nonstationary_p3_r2.yaml"
if ($Batch -eq "bandits") {
    $scenarioConfig = "$FleetPyRoot\experiments\configs\daily10000_logit_switch_f400_p3_r2_bandits_50x6.csv"
    $batchName = "NONSTATIONARY_P3_R2_BANDITS_50x6"
} else {
    $scenarioConfig = "$FleetPyRoot\experiments\configs\daily10000_logit_switch_f400_p3_r2_traditional_etc_50x4.csv"
    $batchName = "NONSTATIONARY_P3_R2_ETC_50x4"
}

$arguments = @(
    "$releaseRoot\run\run_checkpointed_scenarios.py",
    "batch",
    "--constant-config", $constantConfig,
    "--scenario-config", $scenarioConfig,
    "--batch-name", $batchName,
    "--group-column", "priority_group",
    "--max-workers", $Workers,
    "--memory-guard-gib", $MemoryGuardGiB,
    "--launch-stagger-seconds", "3",
    "--network-matrix-mode", "mmap",
    "--log-level", "info",
    "--continue-after-failure"
)
if ($Execute) { $arguments += "--execute" }
& $Python @arguments
exit $LASTEXITCODE
