# saqqarleq-fjord-les

Internal-wave-permitting large-eddy simulation of the Saqqarleq (Sarqardleq) glacial fjord,
West Greenland, built on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) 0.109
(Julia >= 1.10). This is the fjord model behind Sanchez et al., *Observations of a Near-Plume
Internal Tide in a Greenlandic Glacial Fjord*: a rotating, nonhydrostatic run over realistic
Saqqarleq bathymetry, forced by a subglacial-discharge plume that enters the fjord as an outflow
at its neutral-buoyancy depth (the near-terminus rising plume is handled by a separate plume LES,
Zhao et al. 2024).

One script, `iceplume.jl`, runs every scenario through command-line flags. The domain is closed and
the plume is injected as a distributed interior source at its neutral-buoyancy outflow depth, so the
same architecture works with either pressure solver.

## The pressure solver

`iceplume.jl` uses the `ConjugateGradientPoissonSolver` **by default**. It enforces the immersed
no-normal-flow condition in the pressure projection, which removes the spurious near-bathymetry
tracer signal that the FFT solver produced with the `GridFittedBottom` immersed boundary. Tune it
with `--cg_reltol` and `--cg_maxiter`. Validated stepping stably at Δt ~ 15 s on an A100 with
physical velocities (~0.2–0.3 m/s).

Pass `--fft=1` to use Oceananigans' default FFT pressure solve instead. That runs the exact same
closed-domain, interior-source configuration; only the pressure projection changes. It carries the
immersed-bottom tracer artifact by design and is kept as an opt-in cross-check against the earlier
FFT results. (A well-posed closed domain, net divergence ~0, is why FFT is stable here too; the
artifact is solver-intrinsic, not an inflow issue.)

## Why an interior source instead of an open boundary

The `ConjugateGradientPoissonSolver` needs a well-posed (net-zero-divergence) domain, which an
open-boundary net mass flux breaks. So the domain is closed and the plume enters as a distributed
interior source at the glacier face, confined to the outflow layer (`z >= --z_src_bot`) and spread
over `--y_src` metres, balanced by an outflow sponge at the fjord mouth. This is what lets the CG
solver (and its immersed-bottom fix) run at all, and the FFT mode shares it so the two are identical
apart from the solver.

## Scenarios

| scenario     | flags                | description                                            |
|--------------|----------------------|--------------------------------------------------------|
| Control      | (defaults)           | constant subglacial discharge, no tide                 |
| Tide         | `--tide=1`           | constant discharge + external barotropic M2 tide       |
| Varying-SGD  | `--pump=1`           | tidally modulated discharge `Q(t)=Q0(1+A sin(2π t/T))` |
| Both         | `--tide=1 --pump=1`  | tidal discharge and external tide together             |

Default modulation `A = 0.5` (`--pump_amp`, matching `outerpump3`), M2 period `T = 44700 s`
(`--M2_period`), tidal velocity amplitude `0.0168 m/s` (`--tide_amp`, matching `outertide3`).
With `--tide=0 --pump=0` the forcing reproduces the original Control run.

## Running

```bash
# CPU smoke test (laptop; coarse grid, not for science)
julia --project iceplume.jl --arch=cpu --simname=cputest --stop_days=0.02

# On Casper (NCAR): set up the environment once, then submit — see RUN_ON_CASPER.md
JULIA=/path/to/julia ./setup_casper.sh
qsub -v CASE=control submit_casper.sh
qsub -v CASE=tide    submit_casper.sh
qsub -v CASE=pump    submit_casper.sh

# FFT cross-check (same run, default solver instead of CG):
qsub -v CASE=control,EXTRA=--fft=1 submit_casper.sh
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
| `--fft` | `0` | `1` = default FFT pressure solve; `0` = CG (immersed-bottom fix) |
| `--pump_amp` | `0.5` | discharge modulation fraction A (matches outerpump3) |
| `--tide_amp` | `0.0168` | barotropic M2 velocity amplitude [m/s] (matches outertide3) |
| `--M2_period` | `44700` | tidal period [s] (12.42 h) |
| `--y_src` / `--sig_src` | `150` / `60` | interior-source width [m] / relaxation time [s] |
| `--z_src_bot` | `-40` | bottom of the outflow layer the source fills [m] |
| `--stop_days` | `10` | model run length |
| `--Nx --Ny --Nz` | `261 296 100` | grid |
| `--bathymetry` | `bottom.nc` | bathymetry file (var `--bathy_var`, default `bottom`) |
| `--outdir` | `<rundir>/output` | output + checkpoint directory |
| `--cfl` | `0.5` | TimeStepWizard target CFL |
| `--cg_reltol --cg_maxiter` | `1e-4` / `50` | CG Poisson solver tolerance / iteration cap (CG mode) |

## Inputs

`bottom.nc` (committed) is the Saqqarleq bathymetry as negative depths on the model grid; it is
loaded into a 2-D interpolant so the same file works at any resolution.

## Caveats

- The plume enters as an outflow at its neutral-buoyancy depth (~ −17 m), not as a grounding-line
  discharge: the near-terminus rising plume is resolved separately by the plume LES (Zhao et al.
  2024), and this fjord model receives the detached, neutrally buoyant outflow.
- The interior source is EXPERIMENTAL in one respect: `--y_src`/`--sig_src` set how concentrated it
  is, which trades discharge fidelity against timestep. Too concentrated a source drives spurious
  vertical velocity and collapses Δt (the legacy `outer` run forced the discharge into ~1 cell and
  got Δt ≈ 0.22 s). The defaults spread it over a few cells and step at ~15 s.
- Do a CPU smoke test and a short GPU sanity run, then plot the plume placement, before launching a
  production job.

## Reference

Sanchez, R., Straneo, F., Zhao, K., MacKinnon, J. *Observations of a Near-Plume Internal Tide in
a Greenlandic Glacial Fjord* (in prep). Plume forcing from Zhao et al. (2024), *Improved
Parameterizations of Vertical Ice-Ocean Boundary Layers and Melt Rates*, GRL.
