#!/usr/bin/env python3
"""
Evaluation report generator for a voice-conversion pipeline.

What it does
- Loads one or more evaluation JSON objects (single dict, {"datasets":[...]}, or a list of dicts)
- Writes a dataset-level evaluation summary table (CSV + Markdown + console print)
- Writes per-utterance tables for each dataset (CSV)
- Plots key conversion attributes for each dataset (PNG files)

Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn

Usage
  python evaluation_report.py --input evaluation_results.json --outdir evaluation_report
  python evaluation_report.py --input evaluation_results.json --outdir evaluation_report --show

Notes
- The script is defensive: if some fields are missing, it falls back to computing metrics from items where possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Choose a non-interactive backend automatically when running headless (but keep interactive capability if requested).
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- Compatibility shim: seaborn<=0.11 expects a pandas option that was removed in pandas>=2.0
# Without this, seaborn can raise: OptionError: No such keys(s): 'mode.use_inf_as_null'
try:
    from pandas._config import config as _pd_config  # type: ignore
    from pandas._config.config import OptionError as _PandasOptionError  # type: ignore
    try:
        pd.get_option('mode.use_inf_as_null')
    except _PandasOptionError:
        _pd_config.register_option('mode.use_inf_as_null', False)
except Exception:
    # If pandas internals move, we just skip the shim; plots may still work with newer seaborn.
    pass

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _safe_float(x: Any) -> float:
    """Convert x to float or NaN."""
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr, ddof=0))


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "dataset"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_datasets(obj: Any) -> List[Dict[str, Any]]:
    """
    Supported inputs:
      - single dataset dict
      - {"datasets": [dataset_dict, ...]}
      - [dataset_dict, ...]
    """
    if isinstance(obj, list):
        return [d for d in obj if isinstance(d, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("datasets"), list):
            return [d for d in obj["datasets"] if isinstance(d, dict)]
        return [obj]
    raise TypeError(f"Unsupported JSON root type: {type(obj)}")


def dataset_label(ds: Dict[str, Any], idx: int) -> str:
    src = ds.get("source_speaker") or ds.get("conversion_summary", {}).get("source_speaker") or "src"
    tgt = ds.get("target_speaker") or ds.get("conversion_summary", {}).get("target_speaker") or "tgt"
    split = ds.get("split") or ds.get("conversion_summary", {}).get("split") or ""
    base = f"{src}_to_{tgt}"
    if split:
        base += f"_{split}"
    base += f"_{idx+1:02d}"
    return _slugify(base)


def extract_items_df(ds: Dict[str, Any]) -> pd.DataFrame:
    items = ds.get("items") or []
    rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        formant_errors = it.get("formant_errors") or []
        f1e = _safe_float(formant_errors[0]) if len(formant_errors) > 0 else float("nan")
        f2e = _safe_float(formant_errors[1]) if len(formant_errors) > 1 else float("nan")
        f3e = _safe_float(formant_errors[2]) if len(formant_errors) > 2 else float("nan")

        rows.append(
            {
                "idx": i,
                "utt_id": it.get("utt_id"),
                "status": it.get("status"),
                "mcd": _safe_float(it.get("mcd")),
                "f0_correlation": _safe_float(it.get("f0_correlation")),
                "pitch_ratio": _safe_float(it.get("pitch_ratio")),
                "formant_rmse": _safe_float(it.get("formant_rmse")),
                "f1_error": f1e,
                "f2_error": f2e,
                "f3_error": f3e,
                "output_path": it.get("output_path"),
                "target_path": it.get("target_path"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Stable ordering
        df = df.sort_values(["idx"]).reset_index(drop=True)
    return df


def compute_dataset_summary(ds: Dict[str, Any], items_df: pd.DataFrame) -> Dict[str, Any]:
    conv = ds.get("conversion_summary") or {}
    counts = ds.get("counts") or {}

    src = ds.get("source_speaker") or conv.get("source_speaker")
    tgt = ds.get("target_speaker") or conv.get("target_speaker")
    split = ds.get("split") or ""
    generated_at_utc = ds.get("generated_at_utc")

    # Prefer top-level aggregates if present, else compute from items
    mcd = ds.get("mcd") or {}
    f0c = ds.get("f0_correlation") or {}
    frm = ds.get("formant_rmse") or {}

    # Compute from items as fallback and for extra stats
    mcd_vals = items_df["mcd"].tolist() if "mcd" in items_df.columns else []
    f0_vals = items_df["f0_correlation"].tolist() if "f0_correlation" in items_df.columns else []
    pr_vals = items_df["pitch_ratio"].tolist() if "pitch_ratio" in items_df.columns else []
    fr_vals = items_df["formant_rmse"].tolist() if "formant_rmse" in items_df.columns else []

    summary = {
        "source_speaker": src,
        "target_speaker": tgt,
        "split": split,
        "generated_at_utc": generated_at_utc,
        "num_items": int(len(items_df)),
        "evaluated": int(
            counts.get(
                "evaluated",
                int((items_df["status"] == "evaluated").sum()) if (not items_df.empty and "status" in items_df.columns) else 0,
            )
        ),
        "skipped": int(counts.get("skipped", 0)),
        "missing": int(counts.get("missing", 0)),
        "avg_conversion_quality": conv.get("avg_conversion_quality"),
        # Core aggregates
        "mcd_mean": _safe_float(mcd.get("mean")) if isinstance(mcd, dict) else float("nan"),
        "mcd_std": _safe_float(mcd.get("std")) if isinstance(mcd, dict) else float("nan"),
        "f0_correlation_mean": _safe_float(f0c.get("mean")) if isinstance(f0c, dict) else float("nan"),
        "f0_correlation_std": _safe_float(f0c.get("std")) if isinstance(f0c, dict) else float("nan"),
        "formant_rmse_f1": _safe_float(frm.get("f1")) if isinstance(frm, dict) else float("nan"),
        "formant_rmse_f2": _safe_float(frm.get("f2")) if isinstance(frm, dict) else float("nan"),
        "formant_rmse_f3": _safe_float(frm.get("f3")) if isinstance(frm, dict) else float("nan"),
        "formant_rmse_average": _safe_float(frm.get("average")) if isinstance(frm, dict) else float("nan"),
        "pitch_shift_ratio": _safe_float(ds.get("pitch_shift_ratio")),
        # Extra computed stats (useful for debugging)
        "pitch_ratio_mean": _nanmean([_safe_float(v) for v in pr_vals]),
        "pitch_ratio_std": _nanstd([_safe_float(v) for v in pr_vals]),
        "mcd_mean_computed": _nanmean([_safe_float(v) for v in mcd_vals]),
        "f0_correlation_mean_computed": _nanmean([_safe_float(v) for v in f0_vals]),
        "formant_rmse_mean_computed": _nanmean([_safe_float(v) for v in fr_vals]),
    }

    # If top-level aggregates are missing, fill from computed
    if math.isnan(summary["mcd_mean"]):
        summary["mcd_mean"] = summary["mcd_mean_computed"]
    if math.isnan(summary["mcd_std"]):
        summary["mcd_std"] = _nanstd([_safe_float(v) for v in mcd_vals])
    if math.isnan(summary["f0_correlation_mean"]):
        summary["f0_correlation_mean"] = summary["f0_correlation_mean_computed"]
    if math.isnan(summary["f0_correlation_std"]):
        summary["f0_correlation_std"] = _nanstd([_safe_float(v) for v in f0_vals])
    if math.isnan(summary["formant_rmse_average"]):
        summary["formant_rmse_average"] = summary["formant_rmse_mean_computed"]

    return summary


def write_tables(
    summary_df: pd.DataFrame,
    items_by_dataset: List[Tuple[str, pd.DataFrame]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset-level summary
    summary_csv = outdir / "evaluation_summary.csv"
    summary_md = outdir / "evaluation_summary.md"

    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_markdown(summary_md, index=False)

    # Console print (readable in terminals / logs)
    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print("\n=== Evaluation Summary ===")
        print(summary_df.to_string(index=False))

    # Per-utterance tables
    items_dir = outdir / "items"
    items_dir.mkdir(exist_ok=True)
    for label, df in items_by_dataset:
        df_path = items_dir / f"items_{label}.csv"
        df.to_csv(df_path, index=False)


def plot_key_metrics(ds_label: str, ds: Dict[str, Any], items_df: pd.DataFrame, outdir: Path, show: bool) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if items_df.empty:
        print(f"[WARN] No items found for dataset {ds_label}; skipping plots.")
        return

    sns.set_theme()  # seaborn styling, uses matplotlib under the hood

    # Order utterances as they appear
    items_df = items_df.copy()
    items_df["utt_id_str"] = items_df["utt_id"].astype(str)

    # 2x2 key metrics figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) MCD
    sns.lineplot(data=items_df, x="utt_id_str", y="mcd", marker="o", ax=ax1)
    ax1.set_title("MCD by utterance")
    ax1.set_xlabel("utt_id")
    ax1.set_ylabel("mcd")
    ax1.tick_params(axis="x", rotation=45)

    # 2) Pitch ratio (+ expected shift ratio)
    sns.lineplot(data=items_df, x="utt_id_str", y="pitch_ratio", marker="o", ax=ax2)
    psr = _safe_float(ds.get("pitch_shift_ratio"))
    if not math.isnan(psr):
        ax2.axhline(psr, linestyle="--", linewidth=1, label=f"pitch_shift_ratio={psr:.3f}")
        ax2.legend()
    ax2.set_title("Pitch ratio by utterance")
    ax2.set_xlabel("utt_id")
    ax2.set_ylabel("pitch_ratio")
    ax2.tick_params(axis="x", rotation=45)

    # 3) F0 correlation
    sns.barplot(data=items_df, x="utt_id_str", y="f0_correlation", ax=ax3)
    ax3.set_title("F0 correlation by utterance")
    ax3.set_xlabel("utt_id")
    ax3.set_ylabel("f0_correlation")
    ax3.tick_params(axis="x", rotation=45)

    # 4) Formant RMSE
    sns.lineplot(data=items_df, x="utt_id_str", y="formant_rmse", marker="o", ax=ax4)
    ax4.set_title("Formant RMSE by utterance")
    ax4.set_xlabel("utt_id")
    ax4.set_ylabel("formant_rmse")
    ax4.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Key conversion metrics: {ds_label}", fontsize=16)
    key_path = plots_dir / f"key_metrics_{ds_label}.png"
    fig.savefig(key_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    # Formant component errors (F1/F2/F3) if present
    has_formant_errors = items_df[["f1_error", "f2_error", "f3_error"]].notna().any().any()
    if has_formant_errors:
        long_rows = []
        for _, row in items_df.iterrows():
            for f_name, col in [("F1", "f1_error"), ("F2", "f2_error"), ("F3", "f3_error")]:
                v = _safe_float(row[col])
                if not math.isnan(v):
                    long_rows.append({"utt_id": row["utt_id_str"], "formant": f_name, "error_hz": v})
        long_df = pd.DataFrame(long_rows)

        if not long_df.empty:
            fig2, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
            sns.lineplot(data=long_df, x="utt_id", y="error_hz", hue="formant", marker="o", ax=ax)
            ax.set_title(f"Formant errors by utterance: {ds_label}")
            ax.set_xlabel("utt_id")
            ax.set_ylabel("error (Hz)")
            ax.tick_params(axis="x", rotation=45)

            fe_path = plots_dir / f"formant_errors_{ds_label}.png"
            fig2.savefig(fe_path, dpi=200)
            if show:
                plt.show()
            plt.close(fig2)

    # Distributions (helpful for spotting outliers)
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    a1, a2, a3, a4 = axes3.ravel()

    sns.histplot(items_df["mcd"].dropna(), kde=True, ax=a1)
    a1.set_title("MCD distribution")

    sns.histplot(items_df["pitch_ratio"].dropna(), kde=True, ax=a2)
    a2.set_title("Pitch ratio distribution")

    sns.histplot(items_df["f0_correlation"].dropna(), kde=True, ax=a3)
    a3.set_title("F0 correlation distribution")

    sns.histplot(items_df["formant_rmse"].dropna(), kde=True, ax=a4)
    a4.set_title("Formant RMSE distribution")

    fig3.suptitle(f"Metric distributions: {ds_label}", fontsize=16)
    dist_path = plots_dir / f"distributions_{ds_label}.png"
    fig3.savefig(dist_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables and plots from conversion evaluation JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation_results.json (single dict, list, or {'datasets': [...]})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="evaluation_report",
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (also saves PNGs).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)

    # Convenience: if default filename wasn't found, try the common mounted location
    if not in_path.exists() and args.input == "evaluation_results.json":
        mounted = Path("/mnt/data/evaluation_results.json")
        if mounted.exists():
            in_path = mounted

    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obj = load_json(in_path)
    datasets = normalize_datasets(obj)
    if not datasets:
        raise ValueError("No datasets found in the provided JSON.")

    summaries: List[Dict[str, Any]] = []
    items_by_dataset: List[Tuple[str, pd.DataFrame]] = []

    for idx, ds in enumerate(datasets):
        label = dataset_label(ds, idx)
        items_df = extract_items_df(ds)
        summary = compute_dataset_summary(ds, items_df)
        summary["dataset"] = label

        summaries.append(summary)
        items_by_dataset.append((label, items_df))

        plot_key_metrics(label, ds, items_df, outdir, show=args.show)

    # Bring 'dataset' to front for readability
    summary_df = pd.DataFrame(summaries)
    if "dataset" in summary_df.columns:
        cols = ["dataset"] + [c for c in summary_df.columns if c != "dataset"]
        summary_df = summary_df[cols]

    write_tables(summary_df, items_by_dataset, outdir)

    # Optional: cross-dataset comparison plot if multiple datasets are present
    if len(datasets) > 1:
        sns.set_theme()
        comp = summary_df.copy()
        comp_fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        sns.barplot(data=comp, x="dataset", y="mcd_mean", ax=axes[0])
        axes[0].set_title("MCD mean by dataset")
        axes[0].tick_params(axis="x", rotation=45)

        sns.barplot(data=comp, x="dataset", y="formant_rmse_average", ax=axes[1])
        axes[1].set_title("Formant RMSE average by dataset")
        axes[1].tick_params(axis="x", rotation=45)

        sns.barplot(data=comp, x="dataset", y="f0_correlation_mean", ax=axes[2])
        axes[2].set_title("F0 correlation mean by dataset")
        axes[2].tick_params(axis="x", rotation=45)

        comp_path = outdir / "plots" / "dataset_comparison.png"
        comp_fig.savefig(comp_path, dpi=200)
        if args.show:
            plt.show()
        plt.close(comp_fig)

    print(f"\nDone. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()