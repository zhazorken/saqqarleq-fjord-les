#!/usr/bin/env python3
"""
compare_scenarios.py — side-by-side comparison plots across the four saqqarleq-fjord-les scenarios
(control, tide, pump, tidepump).

Produces, into --out (default figures/):
  compare_mooring_v.png / compare_mooring_S.png : virtual-mooring time-depth Hovmöllers
  compare_section_v.png / _S.png / _T.png       : along-fjord (y-z) sections, final snapshot

Each figure is a 2x2 grid of the four scenarios on a shared color scale. Immersed cells (exact 0)
are blanked. Reads <case>_mooring.nc and <case>_xsect.nc from --dir/<case>/.

Usage:
    python compare_scenarios.py --dir output --out figures
Needs: xarray/netCDF4, numpy, matplotlib (NCAR `npl` conda env, or pip install).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

TITLES = {"control": "Control  (steady discharge)", "tide": "Tide  (+ M2)",
          "pump": "Pump  (tidal discharge)", "tidepump": "Tide + Pump"}


def _dim(ds, var, prefix):
    for d in ds.variables[var].dimensions:
        if d.startswith(prefix):
            return d
    return None


def mooring(path, name):
    ds = Dataset(path)
    t = ds.variables["time"][:] / 86400.0
    z = ds.variables[_dim(ds, name, "z")][:]
    a = np.squeeze(np.asarray(ds.variables[name][:]))          # (nt, nz)
    ds.close()
    return t, z, np.where(a == 0.0, np.nan, a)                 # blank immersed


def section(path, name):
    ds = Dataset(path)
    y = ds.variables[_dim(ds, name, "y")][:]
    z = ds.variables[_dim(ds, name, "z")][:]
    a = np.squeeze(np.asarray(ds.variables[name][-1]))         # final snapshot (nz, ny)
    ds.close()
    return y, z, np.where(a == 0.0, np.nan, a)


def _sym(arrs, p=99):
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
    m = np.nanpercentile(np.abs(v), p)
    return -m, m


def _seq(arrs, lo=1, hi=99):
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
    return np.nanpercentile(v, lo), np.nanpercentile(v, hi)


def panel(loader, cases, ddir, name, cmap, sym, title, xlabel, transpose, out, fname,
          ylim=None, xscale=1.0, xlim=None):
    data = []
    for c in cases:
        sub = "mooring" if loader is mooring else "xsect"
        data.append(loader(os.path.join(ddir, c, f"{c}_{sub}.nc"), name))
    arrs = [d[2] for d in data]
    vmin, vmax = (_sym(arrs) if sym else _seq(arrs))
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
    pc = None
    for a, c, (x, z, arr) in zip(ax.ravel(), cases, data):
        field = arr.T if transpose else arr
        pc = a.pcolormesh(x * xscale, z, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        a.set_title(TITLES.get(c, c), fontsize=11)
        a.set_ylabel("z (m)"); a.set_xlabel(xlabel)
        if ylim: a.set_ylim(*ylim)
        if xlim: a.set_xlim(*xlim)
    fig.colorbar(pc, ax=ax, label=title, shrink=0.85)
    fig.suptitle(title, fontsize=13)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, fname)
    fig.savefig(p, dpi=125, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Four-scenario comparison plots.")
    ap.add_argument("--dir", default="output", help="dir holding <case>/<case>_*.nc")
    ap.add_argument("--out", default="figures", help="where to write PNGs")
    ap.add_argument("--cases", default="control,tide,pump,tidepump")
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if os.path.isdir(os.path.join(a.dir, c))]
    print("cases:", cases)

    # mooring Hovmöllers (transpose: field is (nt,nz) -> plot (nz,nt))
    panel(mooring, cases, a.dir, "v", "RdBu_r", True,
          "Virtual mooring time-depth: along-fjord v (m/s)", "time (days)", True, a.out,
          "compare_mooring_v.png", ylim=(-120, 0))
    panel(mooring, cases, a.dir, "S", "viridis", False,
          "Virtual mooring time-depth: salinity S", "time (days)", True, a.out,
          "compare_mooring_S.png", ylim=(-120, 0))

    # along-fjord sections (final snapshot; field already (nz,ny))
    panel(section, cases, a.dir, "v", "RdBu_r", True,
          "Along-fjord section (final): along-fjord v (m/s)", "y — along-fjord (km)", False, a.out,
          "compare_section_v.png", xscale=1e-3, xlim=(0, 5))
    panel(section, cases, a.dir, "S", "viridis", False,
          "Along-fjord section (final): salinity S", "y — along-fjord (km)", False, a.out,
          "compare_section_S.png", xscale=1e-3, xlim=(0, 5))
    panel(section, cases, a.dir, "T", "inferno", False,
          "Along-fjord section (final): temperature T (°C)", "y — along-fjord (km)", False, a.out,
          "compare_section_T.png", xscale=1e-3, xlim=(0, 5))
    print("done")
