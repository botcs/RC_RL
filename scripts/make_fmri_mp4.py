#!/usr/bin/env python3
import argparse
import os
import glob
from typing import List

import numpy as np
import nibabel as nib
import imageio


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_subjects(bids_root: str, subjects: List[str] | None) -> List[str]:
    if subjects:
        return subjects
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(bids_root, "sub-*")) if os.path.isdir(p)])


def load_img(path: str) -> nib.Nifti1Image:
    img = nib.load(path, mmap=True)
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    return img


def pick_slices(zdim: int, n: int = 12) -> List[int]:
    start = max(1, int(0.05 * zdim))
    end = max(start + 1, int(0.95 * zdim))
    idx = np.linspace(start, end - 1, num=min(n, end - start), dtype=int)
    return list(sorted(set(idx)))


def to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def build_mosaic(vol: np.ndarray, slices: List[int], cols: int = 6, gap: int = 2) -> np.ndarray:
    h, w = vol.shape[:2]
    tiles = len(slices)
    cols = max(1, min(cols, tiles))
    rows = int(np.ceil(tiles / cols))
    H = rows * h + (rows - 1) * gap
    W = cols * w + (cols - 1) * gap
    canvas = np.zeros((H, W), dtype=vol.dtype)
    for i, z in enumerate(slices):
        r = i // cols
        c = i % cols
        y = r * (h + gap)
        x = c * (w + gap)
        sl = np.rot90(vol[:, :, z])  # axial view
        canvas[y:y+h, x:x+w] = sl
    return canvas


def write_bold_video(img: nib.Nifti1Image, out_path: str, fps: int = 10, slices_n: int = 12, step: int = 1) -> str:
    dataobj = img.dataobj
    X, Y, Z, T = img.shape
    slices = pick_slices(Z, n=slices_n)
    mid = int(T // 2)
    ref = np.asarray(dataobj[:, :, :, mid], dtype=np.float32)
    vmin = float(np.percentile(ref, 2))
    vmax = float(np.percentile(ref, 98))

    # Precompute tile geometry on first frame
    first_vol = np.asarray(dataobj[:, :, :, 0], dtype=np.float32)
    mosaic0 = build_mosaic(first_vol, slices)
    H, W = mosaic0.shape

    ensure_dir(os.path.dirname(out_path))
    # Prefer libx264; fall back if unavailable
    codecs = ["libx264", "h264", "mpeg4"]
    last_err = None
    for codec in codecs:
        try:
            with imageio.get_writer(out_path, fps=fps, codec=codec) as writer:
                for t in range(0, T, step):
                    vol = np.asarray(dataobj[:, :, :, t], dtype=np.float32)
                    mosaic = build_mosaic(vol, slices)
                    frame = to_uint8(mosaic, vmin, vmax)
                    rgb = np.stack([frame, frame, frame], axis=-1)
                    writer.append_data(rgb)
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode BOLD runs to MP4 mosaics (axial slices over time)")
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--runs-per-sub", type=int, default=1)
    ap.add_argument("--out-dir", default="plots/fmri")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--slices", type=int, default=12)
    ap.add_argument("--step", type=int, default=1, help="Temporal stride (e.g., 2 to halve length)")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.bids_root)
    subs = list_subjects(root, args.subjects)
    if not subs:
        print("No subjects found.")
        return 1

    for sub in subs:
        func = sorted(glob.glob(os.path.join(root, sub, "func", f"{sub}_*_bold.nii.gz")))
        if not func:
            print(f"{sub}: no func runs")
            continue
        for bold in func[: args.runs_per_sub]:
            base = os.path.basename(bold).replace(".nii.gz", "")
            out = os.path.join(args.out_dir, f"{base}_mosaic.mp4")
            print(f"Encoding {sub}: {base} -> {out}")
            img = load_img(bold)
            write_bold_video(img, out, fps=args.fps, slices_n=args.slices, step=args.step)
            print(f"Done: {out}")

    print("All done.")


if __name__ == "__main__":
    raise SystemExit(main())
