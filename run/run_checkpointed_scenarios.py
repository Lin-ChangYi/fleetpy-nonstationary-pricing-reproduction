"""Run FleetPy scenarios as isolated, restartable subprocesses.

The coordinator starts a configurable number of one-core workers. Each scenario is written
to a frozen one-row CSV, receives separate stdout/stderr logs, and is marked
complete only after the expected FleetPy outputs exist. The checkpoint CSV is
replaced atomically after every state change so a later invocation can resume
without rerunning completed scenarios. Optional priority groups order the queue
so every method for an earlier input sample is launched before methods for a
later sample, without leaving workers idle while earlier jobs finish.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import pandas as pd
import psutil


REPO_ROOT = Path(os.environ.get("FLEETPY_ROOT", Path(__file__).resolve().parents[1])).resolve()
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNTIME_CONFIG_DIR = EXPERIMENTS_DIR / "configs" / "runtime"
LOG_DIR = EXPERIMENTS_DIR / "logs"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
FAILED_RUN_ARCHIVE = REPO_ROOT / "archive" / "incomplete_runs"

REQUEST_LOG_FILENAME = "mab_spatial_request_log.csv"
REQUEST_LOG_COLUMNS = {
    "request_id",
    "pricing_episode_start",
    "request_reward_amount",
    "request_reward_observed",
}
REQUIRED_OUTPUTS = (
    "00_config.json",
    "final_state.csv",
    "1_user-stats.csv",
    "2-0_op-stats.csv",
    "standard_eval.csv",
    REQUEST_LOG_FILENAME,
)

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class Job:
    scenario_name: str
    runtime_config: Path
    result_dir: Path
    priority_group: str


@dataclass
class ActiveProcess:
    job: Job
    process: subprocess.Popen
    stdout_handle: IO[str]
    stderr_handle: IO[str]
    stdout_path: Path
    stderr_path: Path
    peak_rss_gib: float = 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _group_value(value) -> str:
    if pd.isna(value):
        raise ValueError("Priority-group values must not be empty.")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _available_gib() -> float:
    return psutil.virtual_memory().available / (1024**3)


def _process_tree_rss_gib(pid: int) -> float:
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running()) / (1024**3)
    except (psutil.Error, OSError):
        return 0.0


def _assert_within(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path is outside the expected directory: {resolved}")


def _study_name(constant_config: Path) -> str:
    if constant_config.parent.name != "scenarios":
        raise ValueError(
            "The constant config must be inside studies/<study>/scenarios so "
            "FleetPy can determine the study name."
        )
    return constant_config.parent.parent.name


def _has_request_log_data(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.reader(handle, strict=True)
            header = next(rows, [])
            if len(set(header)) != len(header) or not REQUEST_LOG_COLUMNS.issubset(header):
                return False

            # Read only the first nonblank record to keep resume checks fast.
            row = next((row for row in rows if any(field.strip() for field in row)), [])
            return len(row) == len(header) and bool(row[header.index("request_id")].strip())
    except (OSError, UnicodeError, csv.Error):
        return False


def _missing_outputs(result_dir: Path) -> list[str]:
    missing = []
    for filename in REQUIRED_OUTPUTS:
        path = result_dir / filename
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or (filename == REQUEST_LOG_FILENAME and not _has_request_log_data(path))
        ):
            missing.append(filename)
    return missing


def _output_status(result_dir: Path) -> tuple[str, list[str]]:
    if not result_dir.exists():
        return "pending", list(REQUIRED_OUTPUTS)
    missing = _missing_outputs(result_dir)
    if missing:
        return "incomplete", missing
    return "completed", []


def _atomic_write_checkpoint(records: dict[str, dict], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records.values())
    sort_column = "launch_order" if "launch_order" in frame.columns else "job_order"
    frame = frame.sort_values(sort_column)
    temp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        frame.to_csv(temp_path, index=False)
        for attempt in range(20):
            try:
                os.replace(temp_path, checkpoint_path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_previous_records(checkpoint_path: Path) -> dict[str, dict]:
    if not checkpoint_path.is_file():
        return {}
    frame = pd.read_csv(checkpoint_path, keep_default_na=False)
    if "scenario_name" not in frame.columns:
        return {}
    return {
        str(row["scenario_name"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def _prepare_jobs(
    constant_config: Path,
    scenario_config: Path,
    batch_name: str,
    scenario_filters: set[str],
    network_matrix_mode: str,
    requested_group_column: str,
) -> tuple[list[Job], dict[str, dict], Path, str | None]:
    source = pd.read_csv(scenario_config, comment="#")
    if "scenario_name" not in source.columns:
        raise ValueError(f"Missing scenario_name column: {scenario_config}")
    source["scenario_name"] = source["scenario_name"].astype(str)
    if source["scenario_name"].duplicated().any():
        duplicates = source.loc[source["scenario_name"].duplicated(), "scenario_name"].tolist()
        raise ValueError(f"Duplicate scenario names: {duplicates}")
    if scenario_filters:
        missing_filters = scenario_filters.difference(set(source["scenario_name"]))
        if missing_filters:
            raise ValueError(f"Unknown scenario names: {sorted(missing_filters)}")
        source = source[source["scenario_name"].isin(scenario_filters)].copy()

    if requested_group_column.lower() in {"none", "off", ""}:
        group_column = None
    elif requested_group_column == "auto":
        group_column = next(
            (candidate for candidate in ("sample_id", "replication_id") if candidate in source.columns),
            None,
        )
    else:
        group_column = requested_group_column
    if group_column is not None and group_column not in source.columns:
        raise ValueError(f"Missing priority-group column {group_column!r}: {scenario_config}")

    study = _study_name(constant_config)
    results_dir = REPO_ROOT / "studies" / study / "results"
    runtime_dir = RUNTIME_CONFIG_DIR / batch_name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{batch_name}.csv"
    previous = _load_previous_records(checkpoint_path)

    jobs: list[Job] = []
    records: dict[str, dict] = {}
    group_orders: dict[str, int] = {}
    for order, (_, source_row) in enumerate(source.iterrows(), start=1):
        scenario_name = str(source_row["scenario_name"])
        priority_group = _group_value(source_row[group_column]) if group_column else "all"
        group_order = group_orders.setdefault(priority_group, len(group_orders) + 1)
        runtime_path = runtime_dir / f"{order:04d}_{_safe_name(scenario_name)}.csv"
        runtime_row = source_row.copy()
        runtime_row["route_output_flag"] = False
        runtime_row["replay_flag"] = False
        runtime_row["show_progress_bar"] = False
        runtime_row["realtime_plot"] = 0
        pd.DataFrame([runtime_row]).to_csv(runtime_path, index=False)

        result_dir = results_dir / scenario_name
        status, missing = _output_status(result_dir)
        old = previous.get(scenario_name, {})
        record = {
            "job_order": order,
            "launch_order": 0,
            "scenario_name": scenario_name,
            "priority_group": priority_group,
            "priority_group_order": group_order,
            "priority_group_column": group_column or "",
            "group_completed_at_utc": old.get("group_completed_at_utc", ""),
            "status": status,
            "constant_config": str(constant_config),
            "source_scenario_config": str(scenario_config),
            "runtime_scenario_config": str(runtime_path),
            "result_dir": str(result_dir),
            "started_at_utc": old.get("started_at_utc", ""),
            "finished_at_utc": old.get("finished_at_utc", ""),
            "exit_code": old.get("exit_code", ""),
            "peak_process_rss_gib": old.get("peak_process_rss_gib", ""),
            "launch_available_gib": old.get("launch_available_gib", ""),
            "stdout_log": old.get("stdout_log", ""),
            "stderr_log": old.get("stderr_log", ""),
            "missing_outputs": ";".join(missing),
            "network_matrix_mode": network_matrix_mode,
            "updated_at_utc": _utc_now(),
        }
        if status == "completed" and not record["finished_at_utc"]:
            record["finished_at_utc"] = _utc_now()
        jobs.append(Job(scenario_name, runtime_path, result_dir, priority_group))
        records[scenario_name] = record

    launch_jobs = sorted(
        jobs,
        key=lambda job: (
            int(records[job.scenario_name]["priority_group_order"]),
            int(records[job.scenario_name]["job_order"]),
        ),
    )
    for launch_order, job in enumerate(launch_jobs, start=1):
        records[job.scenario_name]["launch_order"] = launch_order

    for priority_group in group_orders:
        group_names = [job.scenario_name for job in jobs if job.priority_group == priority_group]
        if all(records[name]["status"] == "completed" for name in group_names):
            completed_at = next(
                (
                    str(records[name]["group_completed_at_utc"])
                    for name in group_names
                    if records[name]["group_completed_at_utc"]
                ),
                _utc_now(),
            )
            for name in group_names:
                records[name]["group_completed_at_utc"] = completed_at
        else:
            for name in group_names:
                records[name]["group_completed_at_utc"] = ""

    _atomic_write_checkpoint(records, checkpoint_path)
    return jobs, records, checkpoint_path, group_column


def _archive_incomplete(job: Job, batch_name: str) -> Path | None:
    status, _ = _output_status(job.result_dir)
    if status != "incomplete":
        return None
    _assert_within(job.result_dir, REPO_ROOT / "studies")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = FAILED_RUN_ARCHIVE / batch_name / f"{_safe_name(job.scenario_name)}_{timestamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(job.result_dir), str(destination))
    return destination


def _launch_job(
    job: Job,
    constant_config: Path,
    batch_name: str,
    log_level: str,
    network_matrix_mode: str,
) -> ActiveProcess:
    batch_log_dir = LOG_DIR / batch_name
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = batch_log_dir / f"{_safe_name(job.scenario_name)}.out.log"
    stderr_path = batch_log_dir / f"{_safe_name(job.scenario_name)}.err.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8", buffering=1)
    stderr_handle = stderr_path.open("w", encoding="utf-8", buffering=1)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--constant-config",
        str(constant_config),
        "--scenario-config",
        str(job.runtime_config),
        "--log-level",
        log_level,
    ]
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENV)
    environment["FLEETPY_NETWORK_MATRIX_MODE"] = network_matrix_mode
    environment["PYTHONUNBUFFERED"] = "1"
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=stdout_handle,
        stderr=stderr_handle,
        creationflags=creation_flags,
    )
    return ActiveProcess(
        job=job,
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _ordered_groups(jobs: list[Job]) -> list[str]:
    return list(dict.fromkeys(job.priority_group for job in jobs))


def _mark_group_completed(
    priority_group: str,
    jobs: list[Job],
    records: dict[str, dict],
    checkpoint_path: Path,
) -> None:
    completed_at = _utc_now()
    for job in jobs:
        if job.priority_group == priority_group:
            records[job.scenario_name]["group_completed_at_utc"] = completed_at
            records[job.scenario_name]["updated_at_utc"] = completed_at
    _atomic_write_checkpoint(records, checkpoint_path)


def _run_batch(args: argparse.Namespace) -> int:
    constant_config = Path(args.constant_config).resolve()
    scenario_config = Path(args.scenario_config).resolve()
    if not constant_config.is_file():
        raise FileNotFoundError(constant_config)
    if not scenario_config.is_file():
        raise FileNotFoundError(scenario_config)

    batch_name = _safe_name(args.batch_name or scenario_config.stem)
    jobs, records, checkpoint_path, group_column = _prepare_jobs(
        constant_config,
        scenario_config,
        batch_name,
        set(args.scenario_name),
        args.network_matrix_mode,
        args.group_column,
    )
    pending = [job for job in jobs if records[job.scenario_name]["status"] != "completed"]
    groups = _ordered_groups(jobs)
    pending.sort(
        key=lambda job: (
            int(records[job.scenario_name]["priority_group_order"]),
            int(records[job.scenario_name]["job_order"]),
        )
    )
    completed_count = len(jobs) - len(pending)
    print(f"Prepared {len(jobs)} jobs: {completed_count} completed, {len(pending)} pending/incomplete.")
    print(f"Checkpoint: {checkpoint_path}")
    if group_column:
        print(
            f"Priority queue: {group_column}; all methods for an earlier group are launched "
            "before methods for a later group."
        )
    else:
        print("Priority queue: source-row order (no sample grouping column found).")
    print(
        "Runtime overrides: route=False, replay=False, progress bar=False, "
        f"realtime plot=0, network matrices={args.network_matrix_mode}."
    )
    if not args.execute:
        print("Preparation only. Pass --execute to run the simulations.")
        return 0
    if not pending:
        print("All selected scenarios are complete.")
        return 0

    active: dict[int, ActiveProcess] = {}
    halt_new_jobs = False
    last_memory_notice = 0.0
    try:
        while pending or active:
            while pending and len(active) < args.max_workers and not halt_new_jobs:
                available = _available_gib()
                if available < args.memory_guard_gib:
                    now = time.monotonic()
                    if now - last_memory_notice >= 60:
                        print(
                            f"Memory guard: {available:.2f} GiB available; waiting for at least "
                            f"{args.memory_guard_gib:.2f} GiB before starting another job.",
                            flush=True,
                        )
                        last_memory_notice = now
                    break

                job = pending.pop(0)
                archived = _archive_incomplete(job, batch_name)
                if archived is not None:
                    print(f"Archived an incomplete prior attempt: {archived}", flush=True)
                launched = _launch_job(
                    job,
                    constant_config,
                    batch_name,
                    args.log_level,
                    args.network_matrix_mode,
                )
                launched.peak_rss_gib = _process_tree_rss_gib(launched.process.pid)
                active[launched.process.pid] = launched
                record = records[job.scenario_name]
                record.update(
                    {
                        "status": "running",
                        "started_at_utc": _utc_now(),
                        "finished_at_utc": "",
                        "exit_code": "",
                        "peak_process_rss_gib": round(launched.peak_rss_gib, 3),
                        "launch_available_gib": round(available, 3),
                        "stdout_log": str(launched.stdout_path),
                        "stderr_log": str(launched.stderr_path),
                        "missing_outputs": ";".join(REQUIRED_OUTPUTS),
                        "network_matrix_mode": args.network_matrix_mode,
                        "updated_at_utc": _utc_now(),
                    }
                )
                _atomic_write_checkpoint(records, checkpoint_path)
                print(
                    f"Started [{len(active)}/{args.max_workers}] {job.scenario_name} "
                    f"(PID {launched.process.pid}, available {available:.2f} GiB)",
                    flush=True,
                )
                if args.launch_stagger_seconds > 0:
                    stagger_deadline = time.monotonic() + args.launch_stagger_seconds
                    while time.monotonic() < stagger_deadline:
                        for running in active.values():
                            running.peak_rss_gib = max(
                                running.peak_rss_gib,
                                _process_tree_rss_gib(running.process.pid),
                            )
                        remaining = stagger_deadline - time.monotonic()
                        if remaining > 0:
                            time.sleep(min(0.5, remaining))

            finished_pids = []
            for pid, running in active.items():
                running.peak_rss_gib = max(running.peak_rss_gib, _process_tree_rss_gib(pid))
                return_code = running.process.poll()
                if return_code is None:
                    continue
                running.stdout_handle.close()
                running.stderr_handle.close()
                status, missing = _output_status(running.job.result_dir)
                if return_code != 0 or status != "completed":
                    status = "failed"
                    if not args.continue_after_failure:
                        halt_new_jobs = True
                record = records[running.job.scenario_name]
                record.update(
                    {
                        "status": status,
                        "finished_at_utc": _utc_now(),
                        "exit_code": return_code,
                        "peak_process_rss_gib": round(running.peak_rss_gib, 3),
                        "missing_outputs": ";".join(missing),
                        "updated_at_utc": _utc_now(),
                    }
                )
                _atomic_write_checkpoint(records, checkpoint_path)
                print(
                    f"Finished {running.job.scenario_name}: status={status}, exit={return_code}, "
                    f"peak RSS={running.peak_rss_gib:.2f} GiB. Checkpoint saved.",
                    flush=True,
                )
                finished_pids.append(pid)
            for pid in finished_pids:
                del active[pid]

            for priority_group in groups:
                group_names = [
                    job.scenario_name for job in jobs if job.priority_group == priority_group
                ]
                already_marked = all(
                    bool(records[name]["group_completed_at_utc"]) for name in group_names
                )
                if not already_marked and all(
                    records[name]["status"] == "completed" for name in group_names
                ):
                    _mark_group_completed(priority_group, jobs, records, checkpoint_path)
                    print(
                        f"Priority group {priority_group} completed across all methods. "
                        "Checkpoint is ready for incremental analysis.",
                        flush=True,
                    )

            if halt_new_jobs and not active:
                print("Stopped before launching more jobs because a scenario failed.", flush=True)
                break
            if active or pending:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Interrupt received; terminating active workers. Completed results remain saved.", flush=True)
        for running in active.values():
            running.process.terminate()
        for running in active.values():
            try:
                running.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                running.process.kill()
            running.stdout_handle.close()
            running.stderr_handle.close()
            status, missing = _output_status(running.job.result_dir)
            records[running.job.scenario_name].update(
                {
                    "status": status if status == "completed" else "interrupted",
                    "finished_at_utc": _utc_now(),
                    "exit_code": running.process.returncode,
                    "peak_process_rss_gib": round(running.peak_rss_gib, 3),
                    "missing_outputs": ";".join(missing),
                    "updated_at_utc": _utc_now(),
                }
            )
        _atomic_write_checkpoint(records, checkpoint_path)
        return 130

    failed = sum(record["status"] != "completed" for record in records.values())
    completed = len(records) - failed
    print(f"Batch status: {completed}/{len(records)} completed; {failed} not completed.")
    return 1 if failed else 0


def _run_worker(args: argparse.Namespace) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from run_examples import run_scenarios

    run_scenarios(
        args.constant_config,
        args.scenario_config,
        n_parallel_sim=1,
        n_cpu_per_sim=1,
        log_level=args.log_level,
        keep_old=False,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch", help="Prepare or execute a restartable scenario batch.")
    batch.add_argument("--constant-config", required=True)
    batch.add_argument("--scenario-config", required=True)
    batch.add_argument("--batch-name")
    batch.add_argument("--scenario-name", action="append", default=[])
    batch.add_argument(
        "--group-column",
        default="auto",
        help=(
            "Sample-priority column. 'auto' uses sample_id, then replication_id, when present. "
            "Earlier groups are launched first without waiting for completion. Use 'none' to "
            "preserve source-row order."
        ),
    )
    batch.add_argument("--max-workers", type=int, choices=range(1, 13), default=8)
    batch.add_argument(
        "--memory-guard-gib",
        type=float,
        default=1.5,
        help="Do not start another worker below this available-memory threshold.",
    )
    batch.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    batch.add_argument("--poll-seconds", type=float, default=5.0)
    batch.add_argument(
        "--network-matrix-mode",
        choices=["array", "mmap"],
        default="array",
        help="Use fast compact NumPy arrays by default; mmap is a lower-memory fallback.",
    )
    batch.add_argument("--log-level", choices=["verbose", "debug", "info", "warning"], default="warning")
    batch.add_argument("--continue-after-failure", action="store_true")
    batch.add_argument(
        "--execute",
        action="store_true",
        help="Actually run pending jobs. Without this flag the command only prepares and checkpoints them.",
    )

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--constant-config", required=True)
    worker.add_argument("--scenario-config", required=True)
    worker.add_argument("--log-level", default="warning")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "worker":
        return _run_worker(args)
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
