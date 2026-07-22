#!/usr/bin/env python3
"""
plot_quicklook.py — fast look at a saqqarleq-fjord-les run's NetCDF output.

Makes three PNGs from the outputs of one scenario:
  <prefix>_mooring.png : time-depth Hovmoller of v, T, S at the virtual mooring
                         (the internal-tide view — compare with Fig 2/3 of the paper)
  <prefix>_xsect.png   : last-snapshot along-fjord (y-z) section of v, T, S, vorticity
  <prefix>_face.png    : last-snapshot plan-view (x-y) speed on a mid-depth slice

Usage:
    python plot_quicklook.py control --dir output/control
    python plot_quicklook.py tide    --dir output/tide --out figures

Needs: xarray, numpy, matplotlib, netCDF4 (all in NCAR's `npl` conda env, or `pip install`).
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

# domain-appropriate colormaps (sequential for T/S/speed, diverging & 0-centered for velocity/vorticity)
CMAP = dict(T="inferno", S="viridis", speed="viridis",
            u="RdBu_r", v="RdBu_r", w="RdBu_r")
DIVERGING = {"u", "v", "w", "ω_y", "omega_y", "vorticity"}


def dim_like(da, prefix):
    for d in da.dims:
        if d.startswith(prefix):
            return d
    return None


def getvar(ds, *names):
    for n in names:
        if n in ds:
            return ds[n], n
    return None, None


def squeeze_point(da):
    """Drop length-1 dims left over from single-index slices (mooring x,y; xsect x; face z),
    but never drop 'time'."""
    drop = [d for d in da.dims if da.sizes[d] == 1 and not d.startswith("time")]
    return da.squeeze(dim=drop, drop=True) if drop else da


def limits(a, diverging):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    if diverging:
        m = np.nanpercentile(np.abs(a), 98) or 1.0
        return -m, m
    return np.nanpercentile(a, 2), np.nanpercentile(a, 98)


def open1(path):
    return xr.open_dataset(path, decode_times=False)


def plot_mooring(prefix, d, out):
    f = os.path.join(d, f"{prefix}_mooring.nc")
    if not os.path.exists(f):
        print(f"  (no {os.path.basename(f)})"); return
    ds = open1(f)
    t = ds["time"].values / 86400.0                      # days
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, ("v", "T", "S")):
        da, _ = getvar(ds, name)
        if da is None:
            ax.set_visible(False); continue
        da = squeeze_point(da)                            # drop singleton x,y from the point slice
        zdim = dim_like(da, "z")
        z = ds[zdim].values
        arr = da.transpose("time", zdim).values          # (nt, nz)
        vmin, vmax = limits(arr, name in DIVERGING)
        pc = ax.pcolormesh(t, z, arr.T, cmap=CMAP.get(name, "viridis"),
                           vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(pc, ax=ax, label=name)
        ax.set_ylabel("z (m)")
    axes[-1].set_xlabel("time (days)")
    axes[0].set_title(f"{prefix}: mooring time-depth")
    fig.tight_layout(); p = os.path.join(out, f"{prefix}_mooring.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  wrote {p}")


def plot_xsect(prefix, d, out):
    f = os.path.join(d, f"{prefix}_xsect.nc")
    if not os.path.exists(f):
        print(f"  (no {os.path.basename(f)})"); return
    ds = open1(f).isel(time=-1)
    names = [n for n in ("v", "T", "S", "ω_y", "omega_y") if n in ds]
    names = names[:4]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 3*len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        da = squeeze_point(ds[name])                     # drop singleton x from the along-fjord slice
        ydim, zdim = dim_like(da, "y"), dim_like(da, "z")
        y, z = ds[ydim].values, ds[zdim].values
        arr = da.transpose(ydim, zdim).values            # (ny, nz)
        vmin, vmax = limits(arr, name in DIVERGING)
        pc = ax.pcolormesh(y, z, arr.T, cmap="RdBu_r" if name in DIVERGING else CMAP.get(name, "viridis"),
                           vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(pc, ax=ax, label=name)
        ax.set_ylabel("z (m)")
    axes[-1, 0].set_xlabel("y — along-fjord (m)")
    axes[0, 0].set_title(f"{prefix}: along-fjord section (final snapshot)")
    fig.tight_layout(); p = os.path.join(out, f"{prefix}_xsect.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  wrote {p}")


def plot_face(prefix, d, out):
    # prefer a mid-depth face if present (face3), else the first face file that exists
    cands = [os.path.join(d, f"{prefix}_face3.nc")] + sorted(glob.glob(os.path.join(d, f"{prefix}_face*.nc")))
    f = next((c for c in cands if os.path.exists(c)), None)
    if f is None:
        print(f"  (no {prefix}_face*.nc)"); return
    ds = open1(f).isel(time=-1)
    u, _ = getvar(ds, "u"); v, _ = getvar(ds, "v")
    if v is None:
        print("  (no velocity on face)"); return
    v = squeeze_point(v)                                 # drop singleton z from the depth slice
    xdim, ydim = dim_like(v, "x"), dim_like(v, "y")      # v lives on x_caa, y_afa
    x, y = ds[xdim].values, ds[ydim].values
    va = v.transpose(xdim, ydim).values
    # u and v sit on different C-grid axes (u on x_faa, y_aca). Interpolate u onto v's points and
    # combine; if that fails for any reason, fall back to the along-fjord speed |v|.
    speed, label = np.abs(va), "|v| along-fjord (m/s)"
    if u is not None:
        try:
            u = squeeze_point(u)
            uxi, uyi = dim_like(u, "x"), dim_like(u, "y")
            ui = u.interp({uxi: x, uyi: y}).transpose(uxi, uyi).values
            if ui.shape == va.shape:
                speed, label = np.hypot(ui, va), "speed (m/s)"
        except Exception as e:
            print(f"  (u/v regrid failed, showing |v|: {e})")
    fig, ax = plt.subplots(figsize=(7, 8))
    vmin, vmax = limits(speed, False)
    pc = ax.pcolormesh(x, y, speed.T, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(pc, ax=ax, label=label)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.set_title(f"{prefix}: plan-view speed on {os.path.basename(f).replace('.nc','')} (final)")
    fig.tight_layout(); p = os.path.join(out, f"{prefix}_face.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quick-look plots for a saqqarleq-fjord-les run.")
    ap.add_argument("prefix", help="run name / --simname (e.g. control, tide, pump)")
    ap.add_argument("--dir", default=".", help="directory holding <prefix>_*.nc (default: .)")
    ap.add_argument("--out", default=None, help="where to write PNGs (default: same as --dir)")
    a = ap.parse_args()
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    print(f"quick-look for '{a.prefix}' in {a.dir}:")
    plot_mooring(a.prefix, a.dir, out)
    plot_xsect(a.prefix, a.dir, out)
    plot_face(a.prefix, a.dir, out)
    print("done.")
