#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from typing import List, Tuple

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def list_subjects(bids_root: str, subjects: List[str] | None) -> List[str]:
    if subjects:
        return subjects
    subs = sorted(
        [os.path.basename(p) for p in glob.glob(os.path.join(bids_root, "sub-*")) if os.path.isdir(p)]
    )
    return subs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_img(path: str) -> nib.Nifti1Image:
    img = nib.load(path, mmap=True)
    # Reorient to RAS for consistent slicing orientation
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    return img


def compute_summary_images(img: nib.Nifti1Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataobj = img.dataobj  # lazy array-like
    # mean/std over time
    mean_img = np.asarray(np.mean(dataobj, axis=-1), dtype=np.float32)
    std_img = np.asarray(np.std(dataobj, axis=-1), dtype=np.float32)
    # middle volume (nearest to T/2)
    T = img.shape[-1]
    mid_vol = np.asarray(dataobj[..., T // 2], dtype=np.float32)
    return mean_img, std_img, mid_vol


def pick_slices(vol: np.ndarray, n_slices: int = 12) -> List[int]:
    z_dim = vol.shape[2]
    # pick evenly spaced slices avoiding edges
    start = max(1, int(0.05 * z_dim))
    end = max(start + 1, int(0.95 * z_dim))
    z_indices = np.linspace(start, end - 1, num=min(n_slices, end - start), dtype=int)
    return list(sorted(set(z_indices)))


def plot_mosaic(vol: np.ndarray, title: str, out_path: str, cmap: str = "gray", vmin=None, vmax=None) -> None:
    slices = pick_slices(vol)
    n = len(slices)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n:
            z = slices[i]
            ax.imshow(np.rot90(vol[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(img: nib.Nifti1Image, out_path: str, title: str, mask: np.ndarray | None = None) -> None:
    dataobj = img.dataobj
    if mask is None:
        mean_img = np.asarray(np.mean(dataobj, axis=-1), dtype=np.float32)
        thr = np.percentile(mean_img, 80) * 0.2
        mask = mean_img > thr
    T = img.shape[-1]
    # compute mean signal per timepoint within mask (memory efficient)
    ts = np.zeros(T, dtype=np.float64)
    # iterate blocks along time to reduce memory spikes
    block = max(1, min(50, T))
    for start in range(0, T, block):
        end = min(T, start + block)
        block_data = np.asarray(dataobj[..., start:end], dtype=np.float32)
        # average over spatial dims with mask
        masked = block_data[mask, :]
        ts[start:end] = masked.mean(axis=0)

    tr = None
    try:
        z = img.header.get_zooms()
        if len(z) >= 4:
            tr = float(z[3])
    except Exception:
        pass
    x = np.arange(T) * (tr if tr else 1.0)

    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(x, ts, lw=1.0)
    ax.set_xlabel("Time (s)" if tr else "Volume")
    ax.set_ylabel("Global mean signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, ls=":")
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_img(img: nib.Nifti1Image) -> str:
    shape = img.shape
    zooms = img.header.get_zooms()
    tr = zooms[3] if len(zooms) > 3 else None
    return f"shape={shape}, zooms={zooms}, TR={tr}"


def visualize_func(bold_path: str, out_dir: str) -> List[str]:
    img = load_img(bold_path)
    mean_img, std_img, mid_vol = compute_summary_images(img)

    base = os.path.basename(bold_path).replace(".nii.gz", "")
    outs = []
    # scale for display
    vmin = np.percentile(mean_img, 2)
    vmax = np.percentile(mean_img, 98)
    plot_mosaic(mean_img, f"Mean BOLD: {base}", os.path.join(out_dir, f"{base}_mean.png"), vmin=vmin, vmax=vmax)
    outs.append(os.path.join(out_dir, f"{base}_mean.png"))

    smin = np.percentile(std_img, 2)
    smax = np.percentile(std_img, 98)
    plot_mosaic(std_img, f"STD BOLD: {base}", os.path.join(out_dir, f"{base}_std.png"), vmin=smin, vmax=smax)
    outs.append(os.path.join(out_dir, f"{base}_std.png"))

    mvmin = np.percentile(mid_vol, 2)
    mvmax = np.percentile(mid_vol, 98)
    plot_mosaic(mid_vol, f"Middle volume: {base}", os.path.join(out_dir, f"{base}_mid.png"), vmin=mvmin, vmax=mvmax)
    outs.append(os.path.join(out_dir, f"{base}_mid.png"))

    # time series
    plot_timeseries(img, os.path.join(out_dir, f"{base}_timeseries.png"), title=f"Global mean signal: {base}")
    outs.append(os.path.join(out_dir, f"{base}_timeseries.png"))

    return outs


def visualize_anat(t1_path: str, out_dir: str) -> List[str]:
    img = load_img(t1_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    vmin = np.percentile(vol, 2)
    vmax = np.percentile(vol, 98)
    base = os.path.basename(t1_path).replace(".nii.gz", "")
    out = os.path.join(out_dir, f"{base}_mosaic.png")
    plot_mosaic(vol, f"T1w: {base}", out, vmin=vmin, vmax=vmax)
    return [out]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize BIDS fMRI dataset: mean/std/mid-volume mosaics and global timeseries.")
    parser.add_argument("--bids-root", required=True, help="Path to BIDS root (e.g., ~/Downloads/ds004323-download)")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (e.g., sub-01 sub-02). Defaults to all.")
    parser.add_argument("--runs-per-sub", type=int, default=2, help="Max functional runs per subject to visualize (default: 2)")
    parser.add_argument("--out-dir", default="plots/fmri", help="Output directory for figures")
    args = parser.parse_args(argv)

    bids_root = os.path.expanduser(args.bids_root)
    if not os.path.isdir(bids_root):
        print(f"Error: BIDS root not found: {bids_root}", file=sys.stderr)
        return 2

    subs = list_subjects(bids_root, args.subjects)
    if not subs:
        print("No subjects found.", file=sys.stderr)
        return 1

    print(f"Found {len(subs)} subjects: {', '.join(subs)}")
    ensure_dir(args.out_dir)

    for sub in subs:
        sub_dir = os.path.join(bids_root, sub)
        # Anatomical
        t1s = sorted(glob.glob(os.path.join(sub_dir, "anat", f"{sub}_T1w.nii.gz")))
        if t1s:
            outs = visualize_anat(t1s[0], args.out_dir)
            print(f"[anat] {sub}: saved {', '.join(os.path.basename(o) for o in outs)}")

        # Functional
        bolds = sorted(glob.glob(os.path.join(sub_dir, "func", f"{sub}_*_bold.nii.gz")))
        if not bolds:
            print(f"[func] {sub}: no BOLD runs found.")
            continue

        print(f"[func] {sub}: {len(bolds)} runs found. Visualizing up to {args.runs_per_sub}...")
        for i, bold in enumerate(bolds[: args.runs_per_sub]):
            img = load_img(bold)
            print(f"  - {os.path.basename(bold)} :: {summarize_img(img)}")
            outs = visualize_func(bold, args.out_dir)
            print(f"    saved: {', '.join(os.path.basename(o) for o in outs)}")

    print(f"Done. Figures in: {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

