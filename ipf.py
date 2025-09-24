#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IPF-based 3D reconstruction for a list of genes.

Inputs:
  - Fitted 1D profiles per axis (CSV): AP, DV, LML
  - Normalized 1D profiles per axis (CSV) for total-count scaling
  - Slice-volume CSVs per axis
  - 3D mask as .npy
  - Gene list (.txt or .csv, one name per line)

Output:
  - .npz with T: (n_genes, Nx, Ny, Nz) and genes: list of gene names
  - Prints basic stats to stdout

Usage example:
  python run_ipf_recon.py \
    --fitted-ap /path/fittedAP.csv \
    --fitted-dv /path/fittedDV.csv \
    --fitted-lml /path/fittedLML.csv \
    --norm-ap /path/AP_RPMnormExpression1D.csv \
    --norm-dv /path/DV_RPMnormExpression1D.csv \
    --norm-lml /path/LML_RPMnormExpression1D.csv \
    --vol-ap /path/sliceVolAP.csv \
    --vol-dv /path/sliceVolDV.csv \
    --vol-lml /path/sliceVolLML.csv \
    --mask /path/mask_ipf.npy \
    --genes /path/genes.txt \
    --out /path/ipf_tensors_all_genes.npz
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ---------------------- IPF utilities ---------------------- #

def make_base_seed(mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Create a base seed with tiny positive mass in all mask voxels and normalized to 1.
    mask: 3D binary/boolean array (nonzeros are allowed; >0 counts as inside).
    """
    M = (mask > 0)
    X = M.astype(np.float64)
    # add epsilon to masked voxels so no slice is exactly zero at start
    X[M] += eps
    X /= X.sum()
    return X


def seed_for_gene(base_seed: np.ndarray, total: float) -> np.ndarray:
    """Scale the base seed to the desired total counts for the gene."""
    return base_seed * float(total)


def ipf_3d(seed: np.ndarray,
           target_x: np.ndarray,
           target_y: np.ndarray,
           target_z: np.ndarray,
           M: np.ndarray | None = None,
           precision: float = 1e-10,
           max_iters: int = 10000) -> np.ndarray:
    """
    3D IPF to match given 1D marginals along each axis.

    Axis mapping (to match your earlier code/comments):
      - X-slices (axis 0)  <- DV marginal
      - Y-slices (axis 1)  <- LML marginal
      - Z-slices (axis 2)  <- AP marginal
    """
    X = seed.astype(np.float64, copy=True)
    if M is not None:
        M = (M > 0)
        X *= M

    prev_x = X.sum(axis=(1, 2))
    prev_y = X.sum(axis=(0, 2))
    prev_z = X.sum(axis=(0, 1))

    for _ in range(max_iters):
        # match X-slices (DV)
        for i in range(X.shape[0]):
            s = X[i].sum()
            if s > 0:
                X[i] *= (target_x[i] / s)
        if M is not None:
            X *= M

        # match Y-slices (LML)
        for j in range(X.shape[1]):
            s = X[:, j].sum()
            if s > 0:
                X[:, j] *= (target_y[j] / s)
        if M is not None:
            X *= M

        # match Z-slices (AP)
        for k in range(X.shape[2]):
            s = X[:, :, k].sum()
            if s > 0:
                X[:, :, k] *= (target_z[k] / s)
        if M is not None:
            X *= M

        new_x = X.sum(axis=(1, 2))
        new_y = X.sum(axis=(0, 2))
        new_z = X.sum(axis=(0, 1))
        if (np.allclose(new_x, prev_x, atol=precision) and
            np.allclose(new_y, prev_y, atol=precision) and
            np.allclose(new_z, prev_z, atol=precision)):
            break
        prev_x, prev_y, prev_z = new_x, new_y, new_z

    return np.nan_to_num(X)


# ---------------------- I/O helpers ---------------------- #

def read_series_csv(path: str) -> pd.Series:
    """Read a 1-column CSV and return the first column as a pandas Series."""
    df = pd.read_csv(path)
    return df.iloc[:, 0]


def load_fitted_with_index(fitted_csv: str, index_like: pd.Index) -> pd.DataFrame:
    """
    Load fitted CSV (no gene index column) and insert the gene index from a reference
    (e.g. normalized file index), then set as index.
    """
    df = pd.read_csv(fitted_csv)
    df.insert(0, 'gene_name', index_like.to_list())
    df = df.set_index('gene_name')
    return df


# ---------------------- Main pipeline ---------------------- #

def main(args: argparse.Namespace) -> None:
    # Load normalization tables (they contain gene index we re-use for fitted files)
    DV_axis_norm = pd.read_csv(args.norm_dv, index_col=0)
    PA_axis_norm = pd.read_csv(args.norm_ap, index_col=0)
    LML_axis_norm = pd.read_csv(args.norm_lml, index_col=0)

    # Load fitted 1D marginals and attach gene index
    DV_axis_fitted = load_fitted_with_index(args.fitted_dv, DV_axis_norm.index)
    PA_axis_fitted = load_fitted_with_index(args.fitted_ap, PA_axis_norm.index)
    LML_axis_fitted = load_fitted_with_index(args.fitted_lml, LML_axis_norm.index)

    # Load slice volumes (as series)
    volume_DV = read_series_csv(args.vol_dv).astype(np.float64)
    volume_PA = read_series_csv(args.vol_ap).astype(np.float64)
    volume_LML = read_series_csv(args.vol_lml).astype(np.float64)

    # Load mask (3D)
    mask = np.load(args.mask)
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {mask.shape}")

    # Load genes list
    # Supports .txt/.csv one name per line
    with open(args.genes, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]

    # Sanity checks: marginals lengths vs mask dimensions
    Nx, Ny, Nz = mask.shape  # map to DV, LML, AP respectively (per your code)
    if len(volume_DV) != Nx:
        raise ValueError(f"DV length {len(volume_DV)} != mask.shape[0] {Nx}")
    if len(volume_LML) != Ny:
        raise ValueError(f"LML length {len(volume_LML)} != mask.shape[1] {Ny}")
    if len(volume_PA) != Nz:
        raise ValueError(f"AP length {len(volume_PA)} != mask.shape[2] {Nz}")

    # Build base seed once
    base_seed = make_base_seed(mask, eps=args.eps)

    tensors_ipf = []
    kept_genes = []

    # Precompute volume normalizers
    volDV = (volume_DV / volume_DV.sum()).to_numpy()
    volLML = (volume_LML / volume_LML.sum()).to_numpy()
    volPA = (volume_PA / volume_PA.sum()).to_numpy()

    # Iterate genes
    for gene in genes:
        if (gene not in DV_axis_fitted.index
                or gene not in LML_axis_fitted.index
                or gene not in PA_axis_fitted.index):
            # silently skip or log
            print(f"[warn] gene not found in all fitted tables: {gene}", file=sys.stderr)
            continue
        if (gene not in DV_axis_norm.index
                or gene not in LML_axis_norm.index
                or gene not in PA_axis_norm.index):
            print(f"[warn] gene not found in all normalized tables: {gene}", file=sys.stderr)
            continue

        # Volume-correct fitted profiles (convert to float64 arrays)
        tx = DV_axis_fitted.loc[gene].to_numpy(dtype=np.float64) * volDV   # DV -> axis 0
        ty = LML_axis_fitted.loc[gene].to_numpy(dtype=np.float64) * volLML # LML -> axis 1
        tz = PA_axis_fitted.loc[gene].to_numpy(dtype=np.float64) * volPA   # AP -> axis 2

        # Total counts per gene from normalized tables (mean of totals across axes)
        av_gene_counts = np.mean([
            PA_axis_norm.loc[gene].to_numpy().sum(),
            DV_axis_norm.loc[gene].to_numpy().sum(),
            LML_axis_norm.loc[gene].to_numpy().sum()
        ])

        # Equalize totals across axes to av_gene_counts
        # (protect against zeros)
        if tx.sum() > 0: tx *= av_gene_counts / tx.sum()
        if ty.sum() > 0: ty *= av_gene_counts / ty.sum()
        if tz.sum() > 0: tz *= av_gene_counts / tz.sum()

        # Seed scaled to the gene total
        X0 = seed_for_gene(base_seed, av_gene_counts)

        # Run IPF
        Vg = ipf_3d(X0, tx, ty, tz, M=mask, precision=args.precision, max_iters=args.max_iters)

        # Apply mask and store (float32 to save space)
        Vg = (Vg * (mask > 0)).astype(np.float32)
        tensors_ipf.append(Vg)
        kept_genes.append(gene)

    if not tensors_ipf:
        raise RuntimeError("No genes were reconstructed. Check gene names and inputs.")

    T_ipf = np.stack(tensors_ipf, axis=0)  # shape (G, Nx, Ny, Nz)

    # Save
    np.savez_compressed(args.out, T=T_ipf, genes=np.asarray(kept_genes, dtype=object))

    print(f"[done] saved: {args.out}")
    print(f"  T shape: {T_ipf.shape}  (genes, Nx, Ny, Nz) = ({len(kept_genes)}, {Nx}, {Ny}, {Nz})")
    print(f"  genes kept: {len(kept_genes)} / {len(genes)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3D IPF reconstruction from 1D marginals")
    p.add_argument("--fitted-ap",  required=True, help="CSV with fitted AP profiles (columns are slices; rows are genes)")
    p.add_argument("--fitted-dv",  required=True, help="CSV with fitted DV profiles")
    p.add_argument("--fitted-lml", required=True, help="CSV with fitted LML profiles")
    p.add_argument("--norm-ap",    required=True, help="CSV with normalized AP profiles (has gene index)")
    p.add_argument("--norm-dv",    required=True, help="CSV with normalized DV profiles (has gene index)")
    p.add_argument("--norm-lml",   required=True, help="CSV with normalized LML profiles (has gene index)")
    p.add_argument("--vol-ap",     required=True, help="CSV with AP slice volumes (1 column)")
    p.add_argument("--vol-dv",     required=True, help="CSV with DV slice volumes (1 column)")
    p.add_argument("--vol-lml",    required=True, help="CSV with LML slice volumes (1 column)")
    p.add_argument("--mask",       required=True, help="NumPy .npy 3D mask (Nx, Ny, Nz)")
    p.add_argument("--genes",      required=True, help="Text/CSV file with one gene name per line")
    p.add_argument("--out",        required=True, help="Output .npz path")
    p.add_argument("--precision",  type=float, default=1e-10, help="IPF convergence atol")
    p.add_argument("--max-iters",  type=int,   default=10000, help="IPF max iterations")
    p.add_argument("--eps",        type=float, default=1e-9,  help="epsilon added to mask voxels in seed")
    args = p.parse_args()
    main(args)
