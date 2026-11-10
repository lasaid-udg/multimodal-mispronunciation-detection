from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich import print

from .analysis import DEFAULT_METRICS, analyze_run
from .config import apply_overrides, load_config
from .data import audit_dataframe, read_csv
from .experiment import run_experiment
from .models import SUPPORTED_MODELS
from .multiplicity import apply_fdr
from .outputs import save_json

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def audit(
    config: Path = typer.Option(Path("configs/final.yaml"), "--config", "-c"),
    output: Path | None = typer.Option(None, "--output"),
):
    """Audit the dataset without training models."""
    cfg = load_config(config)
    root = config.resolve().parent.parent
    p = Path(cfg["data"]["path"]); p = p if p.is_absolute() else root / p
    result = audit_dataframe(read_csv(p), cfg)
    if output:
        save_json(result, output)
    print(result)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/final.yaml"), "--config", "-c"),
    n_jobs: int | None = typer.Option(None, "--n-jobs", "-j"),
    seed: int | None = typer.Option(None, "--seed"),
    outer_folds: int | None = typer.Option(None, "--outer-folds"),
    inner_folds: int | None = typer.Option(None, "--inner-folds"),
    repeats: int | None = typer.Option(None, "--repeats"),
    models: list[str] | None = typer.Option(None, "--model", help=f"Repeatable. Supported: {', '.join(SUPPORTED_MODELS)}"),
    class_weight: str | None = typer.Option(None, "--class-weight", help="tune, balanced, or none"),
    channel: str | None = typer.Option(None, "--channel", help="ch1, ch2, ch3, or ch4"),
    phoneme: str | None = typer.Option(None, "--phoneme", help="all, t, d, theta, or eth (eth = voiced /th/, IPA /ð/)"),
    data_path: Path | None = typer.Option(None, "--data"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    resume: Path | None = typer.Option(None, "--resume", help="Resume an interrupted run directory."),
    no_save_models: bool = typer.Option(False, "--no-save-models", help="Do not save fitted outer-fold models."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages; run.log is still written."),
):
    """Run leakage-safe nested participant-wise CV and save reusable outputs."""
    if repeats is not None and repeats < 1:
        raise typer.BadParameter("--repeats must be >= 1")
    if models:
        unknown = sorted(set(models) - set(SUPPORTED_MODELS))
        if unknown:
            raise typer.BadParameter(f"Unknown models: {unknown}. Supported: {SUPPORTED_MODELS}")
    if class_weight is not None and class_weight not in {"tune", "balanced", "none"}:
        raise typer.BadParameter("--class-weight must be one of: tune, balanced, none")
    if channel is not None and channel.lower() not in {"ch1", "ch2", "ch3", "ch4"}:
        raise typer.BadParameter("--channel must be one of: ch1, ch2, ch3, ch4")
    if phoneme is not None and phoneme.lower() not in {"all", "t", "d", "theta", "eth"}:
        raise typer.BadParameter("--phoneme must be one of: all, t, d, theta, eth")

    cfg = apply_overrides(
        load_config(config), n_jobs=n_jobs, seed=seed, outer_folds=outer_folds,
        inner_folds=inner_folds, repeats=repeats, models=models, data_path=data_path,
        output_root=output_root, save_models=False if no_save_models else None,
        class_weight=class_weight, verbose=not quiet, channel=channel, phoneme=phoneme,
    )
    out = run_experiment(cfg, " ".join(sys.argv), resume_dir=resume)
    print(f"[green]Completed:[/green] {out}")


@app.command("analyze")
def analyze(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    bootstrap: int = typer.Option(2000, "--bootstrap", "-b", min=1, help="Participant-level bootstrap resamples."),
    permutations: int = typer.Option(2000, "--permutations", "-p", min=0, help="Paired participant-level modality permutations."),
    seed: int = typer.Option(2026, "--seed"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Parallel workers for bootstrap/permutation analysis."),
    metric: list[str] | None = typer.Option(None, "--metric", help=f"Repeatable. Defaults: {', '.join(DEFAULT_METRICS)}"),
):
    """Analyze a completed repeated-CV run without retraining.

    Computes repeat summaries, participant-cluster bootstrap 95% CIs, and
    paired A+EMG minus Audio deltas with participant-level permutation p-values.
    """
    out = analyze_run(
        run_dir, bootstrap=bootstrap, permutations=permutations, seed=seed,
        metrics=list(metric) if metric else None, n_jobs=n_jobs,
    )
    print(f"[green]Analysis completed:[/green] {out}")


@app.command("fdr")
def fdr_cmd(
    root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    alpha: float = typer.Option(0.05, "--alpha"),
):
    """Apply BH-FDR separately to the 28 general and 112 phoneme-level BA comparisons."""
    out = apply_fdr(root, output=output, alpha=alpha)
    print(f"[green]FDR table saved:[/green] {out}")


@app.command("validate-data")
def validate_data(
    csv: Path = typer.Option(Path("data/derived/public_dataset.csv"), "--csv", help="CSV to validate."),
    ncols: int = typer.Option(483, "--ncols", help="Expected column count (483=public, 491=original)."),
    expected_sha256: str | None = typer.Option(None, "--expected-sha256", help="Expected SHA-256 (uppercase hex)."),
):
    """Run the 28 automated integrity tests on the public dataset.

    Tests verify SHA-256, row/column counts, subject/word counts, class distribution,
    feature column counts, channel sentinel exclusion, phoneme partitioning, and
    sentinel/inf isolation.  Exit code 0 = all pass; 1 = any failure.

    NOTE: This command is NOT FOR SCIENTIFIC INFERENCE. It validates the
    derived tabular dataset file, not the ML experiment results.
    """
    script = Path(__file__).parent.parent.parent.parent / "scripts" / "validate_dataset.py"
    if not script.exists():
        # Try relative to cwd (project root)
        script = Path("scripts") / "validate_dataset.py"
    if not script.exists():
        print("[red]validate_dataset.py not found. Expected at scripts/validate_dataset.py[/red]")
        raise typer.Exit(code=1)
    cmd = [sys.executable, str(script), "--csv", str(csv), "--ncols", str(ncols)]
    if expected_sha256:
        cmd += ["--expected-sha256", expected_sha256]
    result = subprocess.run(cmd)
    raise typer.Exit(code=result.returncode)


@app.command("smoke")
def smoke_cmd(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c",
                                 help="Smoke config (default: configs/smoke.yaml)."),
    output_root: Path | None = typer.Option(None, "--output-root"),
    n_jobs: int | None = typer.Option(None, "--n-jobs", "-j"),
):
    """Run the pipeline smoke test (NOT FOR SCIENTIFIC INFERENCE).

    Executes a short run (2 outer folds, 2 inner folds, 1 repeat, 3 models,
    CH2/all phonemes) to verify the pipeline works end-to-end after installation.
    Results are NOT comparable to the manuscript.

    Typical runtime: < 60 seconds.
    """
    print("[yellow]SMOKE TEST — results are NOT for scientific inference[/yellow]")
    cfg = apply_overrides(
        load_config(config),
        n_jobs=n_jobs,
        output_root=output_root,
    )
    out = run_experiment(cfg, " ".join(sys.argv))
    print(f"[green]Smoke test completed:[/green] {out}")
    print("[yellow]Reminder: smoke results are NOT comparable to the manuscript.[/yellow]")


@app.command("reproduce")
def reproduce(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config", "-c",
                                 help="Frozen paper config (default: configs/paper.yaml)."),
    channel: str = typer.Option(..., "--channel", help="ch1, ch2, ch3, or ch4"),
    phoneme: str = typer.Option("all", "--phoneme", help="all, t, d, theta, or eth"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    resume: Path | None = typer.Option(None, "--resume", help="Resume an interrupted run directory."),
    n_jobs: int | None = typer.Option(None, "--n-jobs", "-j"),
    no_save_models: bool = typer.Option(False, "--no-save-models", help="Do not save fitted .joblib models."),
    quiet: bool = typer.Option(False, "--quiet"),
):
    """Run one full-protocol scenario (one channel × phoneme combination).

    Uses the frozen paper configuration (5 outer folds, 3 inner folds, 10 repeats,
    7 models, seed=2026, 2000 bootstrap/permutation replicates).

    To reproduce all 20 scenarios, run once for each combination:
      channels: ch1, ch2, ch3, ch4
      phonemes: all, t, d, theta, eth

    Estimated wall time per scenario: 5–15 CPU-hours (varies by channel/model).
    Total for all 20 scenarios: ~50–100 CPU-hours.

    The run can be safely interrupted and resumed with --resume <run_dir>.
    """
    if channel.lower() not in {"ch1", "ch2", "ch3", "ch4"}:
        raise typer.BadParameter("--channel must be one of: ch1, ch2, ch3, ch4")
    if phoneme.lower() not in {"all", "t", "d", "theta", "eth"}:
        raise typer.BadParameter("--phoneme must be one of: all, t, d, theta, eth")

    cfg = apply_overrides(
        load_config(config),
        n_jobs=n_jobs,
        output_root=output_root,
        save_models=False if no_save_models else None,
        verbose=not quiet,
        channel=channel,
        phoneme=phoneme,
    )
    out = run_experiment(cfg, " ".join(sys.argv), resume_dir=resume)
    print(f"[green]Scenario {channel.upper()}/{phoneme} completed:[/green] {out}")
    print(f"[cyan]Next step: md-emg analyze {out}[/cyan]")


if __name__ == "__main__":
    app()
