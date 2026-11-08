"""Plot convergence and ablation results for the RIS-MLP experiments.

This script is independent from ``MLP_Module.py``. It only reads the CSV logs
produced by training and creates publication-style figures.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent

LOG_FILES = {
    (1, 0): "train_log_G1_hl0.csv",
    (1, 1): "train_log_G1_hl1.csv",
    (1, 2): "train_log_G1_hl2.csv",
    (1, 3): "train_log_G1_hl3.csv",
    (2, 3): "train_log_hl3_G2.csv",
    (4, 3): "train_log_hl3_G4.csv",
}

BASELINE_G = 1
BASELINE_HL = 3

COLOR_LOSS = "#1F4E79"
COLOR_VAL_LOSS = "#D97706"
COLOR_RATE = "#2E8B57"


def configure_matplotlib() -> None:
    """Apply a restrained serif style suitable for a paper figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 5.0,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one training log, failing clearly when the file is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment log: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_convergence(path: Path) -> dict[str, list[float]]:
    """Extract the numerical epoch rows used by the convergence figure."""
    rows = read_csv_rows(path)
    epoch_rows = [row for row in rows if row.get("epoch", "").strip().isdigit()]
    epoch_rows.sort(key=lambda row: int(row["epoch"]))

    if not epoch_rows:
        raise ValueError(f"No epoch rows found in {path}")

    series: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_avg_rate": [],
    }
    for row in epoch_rows:
        try:
            series["epoch"].append(float(row["epoch"]))
            series["train_loss"].append(float(row["train_loss"]))
            series["val_loss"].append(float(row["val_loss"]))
            series["val_avg_rate"].append(float(row["val_avg_rate"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid convergence data at epoch {row.get('epoch')} in {path}"
            ) from exc
    return series


def load_test_avg_rate(path: Path) -> float:
    """Read the final normalized test average rate."""
    test_rows = [
        row for row in read_csv_rows(path) if row.get("epoch", "").strip() == "test"
    ]
    if len(test_rows) != 1:
        raise ValueError(
            f"Expected exactly one test row in {path}, found {len(test_rows)}"
        )

    row = test_rows[0]
    try:
        test_avg_rate = float(row["test_avg_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Incomplete test metrics in {path}") from exc
    return test_avg_rate


def style_axis(axis: plt.Axes) -> None:
    """Use inward ticks and a light dashed grid."""
    axis.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.35)
    axis.tick_params(direction="in", length=3.5, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def plot_convergence(log_path: Path, output_dir: Path) -> None:
    """Create Figure 1 with train/validation loss and validation average rate."""
    series = load_convergence(log_path)

    fig, loss_axis = plt.subplots(figsize=(7.2, 4.4))
    metric_axis = loss_axis.twinx()

    train_loss_line = loss_axis.plot(
        series["epoch"],
        series["train_loss"],
        color=COLOR_LOSS,
        linewidth=1.65,
        label="Training loss",
    )[0]
    val_loss_line = loss_axis.plot(
        series["epoch"],
        series["val_loss"],
        color=COLOR_VAL_LOSS,
        linewidth=1.55,
        linestyle="--",
        label="Validation loss",
    )[0]
    rate_line = metric_axis.plot(
        series["epoch"],
        series["val_avg_rate"],
        color=COLOR_RATE,
        linewidth=1.55,
        linestyle="--",
        label="Validation average rate",
    )[0]

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("MSE loss")
    metric_axis.set_ylabel("Normalized average rate")
    loss_axis.set_xlim(series["epoch"][0], series["epoch"][-1])
    loss_axis.set_ylim(bottom=0.0)
    loss_axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    style_axis(loss_axis)
    style_axis(metric_axis)
    loss_axis.legend(
        handles=[train_loss_line, val_loss_line, rate_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#B0B0B0",
    )

    fig.tight_layout()
    png_path = output_dir / "fig1_convergence_G1_hl3.png"
    pdf_path = output_dir / "fig1_convergence_G1_hl3.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_ablation_summary(
    metrics: dict[tuple[int, int], float],
    output_dir: Path,
) -> None:
    """Create Figure 2 with separate depth and width ablation panels."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    marker_rate = "s"

    depth_hl = [0, 1, 2, 3]
    depth_rate = [100.0 * metrics[(1, hl)] for hl in depth_hl]

    depth_axis = axes[0]
    depth_axis.plot(
        depth_hl,
        depth_rate,
        color=COLOR_RATE,
        marker=marker_rate,
        linestyle="--",
        label="Test average rate",
    )
    depth_axis.set_xlabel("Number of hidden layers, hl")
    depth_axis.set_ylabel("Performance (%)")
    depth_axis.set_title("Depth ablation (G = 1)")
    depth_axis.set_xticks(depth_hl)
    depth_axis.set_ylim(0.0, 105.0)

    width_g = [1, 2, 4]
    width_rate = [100.0 * metrics[(g, 3)] for g in width_g]

    width_axis = axes[1]
    width_axis.plot(
        width_g,
        width_rate,
        color=COLOR_RATE,
        marker=marker_rate,
        linestyle="--",
        label="Test average rate",
    )
    width_axis.set_xlabel("Width scaling factor, G")
    width_axis.set_ylabel("Performance (%)")
    width_axis.set_title("Width ablation (hl = 3)")
    width_axis.set_xticks(width_g)
    width_axis.set_ylim(90.0, 101.2)

    for axis in axes:
        style_axis(axis)
        axis.legend(
            loc="best",
            frameon=True,
            framealpha=0.95,
            edgecolor="#B0B0B0",
        )

    fig.tight_layout(w_pad=2.2)
    png_path = output_dir / "fig2_ablation_summary.png"
    pdf_path = output_dir / "fig2_ablation_summary.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    """Read all six logs and generate Figures 1 and 2."""
    configure_matplotlib()

    paths = {key: SCRIPT_DIR / name for key, name in LOG_FILES.items()}
    metrics = {key: load_test_avg_rate(path) for key, path in paths.items()}

    plot_convergence(paths[(BASELINE_G, BASELINE_HL)], SCRIPT_DIR)
    plot_ablation_summary(metrics, SCRIPT_DIR)

    print("\nFinal test metrics used in Figure 2:")
    for (scale_factor, hidden_layers), avg_rate in metrics.items():
        print(
            f"G={scale_factor}, hl={hidden_layers}: "
            f"avg_rate={avg_rate:.6f}"
        )


if __name__ == "__main__":
    main()
