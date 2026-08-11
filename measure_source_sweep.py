#!/usr/bin/env python3
"""
measure_source_sweep.py — measure whether concentrating the plume source raises the near-plume M2
velocity swing at the glacier mooring back toward the old ~0.025, and whether the run stayed stable.

For each <tag>_mooring.nc it:
  1. finds the outflow core (depth of max time-mean |v| after spin-up),
  2. fits the M2 tidal component there by least squares (robust to a non-integer number of periods)
     and reports the swing AMPLITUDE (v ~ A sin(2 pi t / T_M2)); compare to 0.01 (current) / 0.025 (old),
  3. reports stability: peak |v|, any NaNs, and how many days the record actually reached
     (a run that blew up / died early has a short record).

It also builds three figures: an amplitude bar chart with 0.01 and 0.025 reference lines, an overlay
of the outflow-core v(t), and the M2-amplitude-vs-depth profile (shows the layer concentrating).

Run on Casper in the npl env (has netCDF4), or locally after staging the mooring files:
    module load conda; conda activate npl
    python measure_source_sweep.py --runs output/sweep --out figures/sweep
    # optionally overlay your old and current runs:
    python measure_source_sweep.py --runs output/sweep \
        --extra "old=/path/to/outerpump3_mooring.nc" --extra "current=output/pump/pump_mooring.nc" \
        --out figures/sweep
"""
import argparse, glob, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

M2 = 44700.0                      # M2 period [s] (matches the model)
CUR_REF, OLD_REF = 0.010, 0.025   # current (diffuse) and old (concentrated) mooring swings [m/s]


def load_mooring(path):
    """Return (t[s], z[m], v[t,z]) from a mooring column file, detecting the z/time coords robustly."""
    d = Dataset(path)
    v = d.variables["v"]
    zdim = [dn for dn in v.dimensions if dn.lower().startswith("z")]
    tdim = [dn for dn in v.dimensions if dn.lower().startswith("t")]
    if not zdim or not tdim:
        d.close(); raise ValueError(f"{path}: could not find z/time dims in v{ v.dimensions }")
    zname, tname = zdim[0], tdim[0]
    z = np.asarray(d.variables[zname][:], float)
    t = np.asarray(d.variables[tname][:], float)
    V = np.squeeze(np.asarray(v[:], float))          # -> (time, z)
    if V.shape[0] != t.size and V.shape[-1] == t.size:
        V = V.T
    attrs = {k: d.getncattr(k) for k in d.ncattrs()}
    d.close()
    return t, z, V, attrs


def m2_amplitude(t, x):
    """Least-squares amplitude of the M2 sinusoid in x(t): x ~ a sin(w t)+b cos(w t)+c."""
    good = np.isfinite(x)
    if good.sum() < 8:
        return np.nan
    w = 2 * np.pi / M2
    A = np.column_stack([np.sin(w * t[good]), np.cos(w * t[good]), np.ones(good.sum())])
    coef, *_ = np.linalg.lstsq(A, x[good], rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def analyze(path, spinup_days, z_lo, z_hi):
    t, z, V, attrs = load_mooring(path)
    days = (t[-1] - t[0]) / 86400.0 if t.size > 1 else 0.0
    keep = t >= (t[0] + spinup_days * 86400.0)
    if keep.sum() < 8:                       # too short (died early): use whatever we have
        keep = np.ones_like(t, bool)
    tk, Vk = t[keep], V[keep]

    band = (z >= z_lo) & (z <= z_hi)
    mprof = np.nanmean(np.abs(Vk), axis=0)
    core_k = np.nanargmax(np.where(band, mprof, -np.inf))   # outflow core = max time-mean |v| in band
    v_core = Vk[:, core_k]

    amp_z = np.array([m2_amplitude(tk, Vk[:, k]) for k in range(z.size)])
    return dict(
        path=path, z=z, t=t, V=V, tk=tk, v_core=v_core, core_z=float(z[core_k]),
        amp=m2_amplitude(tk, v_core), amp_z=amp_z,
        mean_out=float(np.nanmean(v_core)), vmax=float(np.nanmax(np.abs(V))),
        nnan=int(np.isnan(V).sum()), days=days, attrs=attrs,
    )


def label_for(path):
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.basename(path).replace("_mooring.nc", "")
    return parent if parent.startswith("s") and parent[1:2].isdigit() else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="output/sweep", help="dir searched recursively for *_mooring.nc")
    ap.add_argument("--extra", action="append", default=[], help="label=path extra series (old/current)")
    ap.add_argument("--spinup_days", type=float, default=1.0)
    ap.add_argument("--z_lo", type=float, default=-35.0)
    ap.add_argument("--z_hi", type=float, default=-3.0)
    ap.add_argument("--out", default="figures/sweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    series = []   # (label, path)
    for p in sorted(glob.glob(os.path.join(a.runs, "**", "*_mooring.nc"), recursive=True)):
        series.append((label_for(p), p))
    for e in a.extra:
        if "=" in e:
            lab, p = e.split("=", 1); series.append((lab, p))
    if not series:
        raise SystemExit(f"no *_mooring.nc found under {a.runs} (and no --extra given)")

    rows = []
    for lab, p in series:
        try:
            r = analyze(p, a.spinup_days, a.z_lo, a.z_hi); r["label"] = lab; rows.append(r)
        except Exception as ex:
            print(f"  ! {lab}: {ex}")

    # ---- summary table + CSV ----
    hdr = f"{'config':<14}{'M2 swing':>10}{'vs .025':>9}{'mean out':>10}{'core z':>8}{'peak|v|':>9}{'days':>6}{'NaN':>5}  stable"
    print("\n" + hdr); print("-" * len(hdr))
    csv_rows = []
    for r in rows:
        stable = (r["nnan"] == 0) and (r["vmax"] < 1.0) and (r["days"] >= 0.9 * max(x["days"] for x in rows))
        pct = 100 * r["amp"] / OLD_REF if np.isfinite(r["amp"]) else float("nan")
        print(f"{r['label']:<14}{r['amp']:>10.4f}{pct:>8.0f}%{r['mean_out']:>10.3f}"
              f"{r['core_z']:>8.0f}{r['vmax']:>9.3f}{r['days']:>6.1f}{r['nnan']:>5}  {'yes' if stable else 'NO'}")
        csv_rows.append(dict(config=r["label"], m2_swing=round(r["amp"], 5), pct_of_old=round(pct, 1),
                             mean_outflow=round(r["mean_out"], 4), core_z=round(r["core_z"], 1),
                             peak_v=round(r["vmax"], 4), days=round(r["days"], 2), nnan=r["nnan"],
                             stable=bool(stable)))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    print("\nReference: current diffuse source ~ %.3f, old concentrated source ~ %.3f m/s" % (CUR_REF, OLD_REF))
    print("wrote", os.path.join(a.out, "summary.csv"))

    # ---- fig 1: M2 amplitude bar chart with reference lines ----
    fig, ax = plt.subplots(figsize=(1.4 * len(rows) + 2, 4.2))
    labs = [r["label"] for r in rows]; amps = [r["amp"] for r in rows]
    stab = [(r["nnan"] == 0 and r["vmax"] < 1.0 and r["days"] >= 0.9 * max(x["days"] for x in rows)) for r in rows]
    ax.bar(labs, amps, color=["#2c7fb8" if s else "#d7301f" for s in stab])
    ax.axhline(OLD_REF, color="#238b45", ls="--", lw=1.4, label=f"old concentrated ~{OLD_REF:.3f}")
    ax.axhline(CUR_REF, color="#888888", ls=":", lw=1.4, label=f"current diffuse ~{CUR_REF:.3f}")
    ax.set_ylabel(r"M2 swing amplitude at outflow core  $|v|$  (m s$^{-1}$)")
    ax.set_title("Source-sensitivity: near-plume M2 swing (blue = stable, red = blew up)")
    ax.legend(fontsize=8); plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "m2_amplitude_bars.png"), dpi=200)

    # ---- fig 2: outflow-core v(t) overlay ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for r in rows:
        ax.plot((r["tk"] - r["tk"][0]) / 86400.0, r["v_core"], lw=1.0, label=f"{r['label']} (z={r['core_z']:.0f} m)")
    ax.set_xlabel("time since spin-up (days)"); ax.set_ylabel(r"outflow-core $v$ (m s$^{-1}$)")
    ax.set_title("Mooring outflow-core velocity"); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "vcore_timeseries.png"), dpi=200)

    # ---- fig 3: M2 amplitude vs depth ----
    fig, ax = plt.subplots(figsize=(4.6, 6))
    for r in rows:
        ax.plot(r["amp_z"], r["z"], lw=1.5, label=r["label"])
    ax.axvline(OLD_REF, color="#238b45", ls="--", lw=1.0); ax.axvline(CUR_REF, color="#888888", ls=":", lw=1.0)
    ax.set_ylim(-60, 0); ax.set_xlabel(r"M2 swing amplitude (m s$^{-1}$)"); ax.set_ylabel("depth (m)")
    ax.set_title("M2 swing vs depth\n(sharper/stronger peak = less diffuse)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "m2_amplitude_vs_depth.png"), dpi=200)
    print("wrote 3 figures to", a.out)


if __name__ == "__main__":
    main()
