# saqqarleq-fjord-les

Internal-wave-permitting large-eddy simulation of the Saqqarleq (Sarqardleq) glacial fjord,
West Greenland, built on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) 0.109
(Julia >= 1.10). This is the fjord model behind Sanchez et al., *Observations of a Near-Plume
Internal Tide in a Greenlandic Glacial Fjord*: a rotating, nonhydrostatic run over realistic
Saqqarleq bathymetry, forced at the grounding line by a subglacial-discharge plume derived from
a near-terminus plume LES (Zhao et al. 2024).

One script, `iceplume.jl`, runs all three paper scenarios through command-line flags.

## Scenarios

| scenario     | flags                | description                                            |
|--------------|----------------------|--------------------------------------------------------|
| Control      | (defaults)           | constant subglacial discharge, no tide                 |
| Tide         | `--tide=1`           | constant discharge + external barotropic M2 tide       |
| Varying-SGD  | `--pump=1`           | tidally modulated discharge `Q(t)=Q0(1+A sin(2π t/T))` |
| Both         | `--tide=1 --pump=1`  | tidal discharge and external tide together             |

Default modulation `A = 0.5` (`--pump_amp`, matching `outerpump3`), M2 period `T = 44700 s`
(`--M2_period`), tidal velocity amplitude `0.0168 m/s` (`--tide_amp`, matching `outertide3`).
With `--tide=0 --pump=0` the forcing reproduces the original Control run exactly.

## The pressure solver fix

The earlier version of this run used Oceananigans' default FFT-based pressure solve, which is
inconsistent with the `GridFittedBottom` immersed boundary and produced a spurious tracer signal
near the bathymetry. This version uses `ConjugateGradientPoissonSolver`, which enforces the
immersed no-normal-flow condition in the pressure projection and removes that artifact. Tune it
with `--cg_reltol` and `--cg_maxiter`.

## Running

```bash
# CPU smoke test (laptop; coarse 60³ grid, not for science)
julia --project iceplume.jl --arch=cpu --simname=cputest --stop_days=0.02

# On Casper (NCAR): set up the environment once, then submit
JULIA=/path/to/julia ./setup_casper.sh
qsub -v CASE=control submit_casper.sh
qsub -v CASE=tide    submit_casper.sh
qsub -v CASE=pump    submit_casper.sh
```

Outputs (`*_face*.nc` z-slices, `*_mooring.nc` virtual-mooring column, `*_xsect.nc` along-fjord
section, `*_timeavg.nc` 6 h average) and checkpoints are written to `--outdir` (default
`./output`, kept out of the repo). Runs auto-resume from a checkpoint if resubmitted with the
same `--simname`.

## Key flags

| flag | default | meaning |
|------|---------|---------|
| `--simname` | `control` | output/checkpoint prefix |
| `--arch` | `auto` | `cpu`, `gpu`, or auto-detect |
| `--tide` / `--pump` | `0` / `0` | scenario switches |
| `--pump_amp` | `0.5` | discharge modulation fraction A (matches outerpump3) |
| `--tide_amp` | `0.0168` | barotropic M2 velocity amplitude [m/s] (matches outertide3) |
| `--M2_period` | `44700` | tidal period [s] (12.42 h) |
| `--stop_days` | `10` | model run length |
| `--Nx --Ny --Nz` | `522 604 80` | grid (domain 12937 × 15039 × 200 m) |
| `--bathymetry` | `bottom.nc` | bathymetry file (var `--bathy_var`, default `bottom`) |
| `--outdir` | `<rundir>/output` | output + checkpoint directory |
| `--cg_reltol --cg_maxiter` | `1e-5` / `50` | CG Poisson solver tolerance / iteration cap |

## Inputs

`bottom.nc` (committed) is the Saqqarleq bathymetry as negative depths on the model grid; it is
loaded into a 2-D interpolant so the same file works at any resolution.

## Caveats

- The `--tide` and `--pump` forcings match the `outertide3` / `outerpump3` runs (tide nudges v to
  `0.0168·sin(2π t/44700)` in the deep open region; plume inflow × `(1 + 0.5·sin(2π t/44700))`).
  Those runs feed the plume through a `south` open boundary; this script keeps the reference
  `outer` interior-sponge plume so the Control stays identical, giving the same forced physics by
  a validated route. If you want the exact south-boundary inflow architecture instead, say so.
- Do a CPU smoke test and a short GPU sanity run before launching a production job.

## Reference

Sanchez, R., Straneo, F., Zhao, K., MacKinnon, J. *Observations of a Near-Plume Internal Tide in
a Greenlandic Glacial Fjord* (in prep). Plume forcing from Zhao et al. (2024), *Improved
Parameterizations of Vertical Ice-Ocean Boundary Layers and Melt Rates*, GRL.
