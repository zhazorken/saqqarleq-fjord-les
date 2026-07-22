# Running a scenario on Casper (NCAR)

Production runs `iceplume.jl`: closed domain, interior plume source at neutral buoyancy,
`ConjugateGradientPoissonSolver` by default (the immersed-bottom tracer fix), validated stepping
stably at Δt ~ 15 s on an A100. Add `EXTRA=--fft=1` for the FFT cross-check (same run, default
solver). Below runs the **control** case; swap `CASE=control` for `tide`, `pump`, or `tidepump`.

Environment vs your old `outerpump3`/`outertide3` runs: same ncarenv 23.10 + peak-memusage, but
**A100 instead of v100**, a **juliaup Julia >= 1.10.11** (not the julia/1.10.5 module), and **no
cuda module** (CUDA.jl brings its own toolkit).

## 0. Allocation

`submit_casper.sh` and the interactive command below use `-A UGIT0046`. Everything else matches the
original runs.

## 1. Copy the code to Casper (run on your laptop)

```bash
rsync -avh --exclude output/ --exclude logs/ \
  ~/Desktop/saqqarleq-fjord-les/ \
  kenzhao@data-access.ucar.edu:/glade/work/kenzhao/saqqarleq-fjord-les/
```

This includes `bottom.nc` (the bathymetry). (Alternatively, once the repo is public:
`git clone <url> /glade/work/kenzhao/saqqarleq-fjord-les`.)

## 2. Install Julia and set up the environment (once, Casper login node)

Casper's `julia/1.10.5` module has a broken Pkg resolver (`KeyError "GPUArraysCore"`), so use a
juliaup Julia >= 1.10.11 (what Ovall26 used):

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes
source ~/.bashrc
juliaup add 1.10 && juliaup default 1.10

cd /glade/work/kenzhao/saqqarleq-fjord-les
bash setup_casper.sh
```

`setup_casper.sh` uses `~/.juliaup/bin/julia`, resolves Oceananigans 0.109 into
`/glade/work/kenzhao/.julia`, and precompiles. Do it on a login node (needs network for `Pkg`).
Takes a while the first time. If you had already run the 1.10.5 module, `rm -f Manifest.toml`
first so 1.10.11 resolves cleanly.

## 3. Quick GPU sanity check (a couple of minutes)

Confirms it builds and steps on the GPU with the CG solver before spending queue time.

```bash
qsub -I -A UGIT0046 -q casper -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=a100 -l walltime=00:30:00
# on the compute node:
cd /glade/work/kenzhao/saqqarleq-fjord-les
module purge; module load ncarenv/23.10
export JULIA_DEPOT_PATH=/glade/work/kenzhao/.julia
export CUDA_VISIBLE_DEVICES=0   # Casper hands it out as a GPU UUID this CUDA.jl can't parse
~/.juliaup/bin/julia --project iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01
```

Watch for: "Model built", the CG solver converging, `max|u|` growing to a physical ~0.2–0.3 m/s
(not NaN, not tens of m/s). Then `exit`.

## 4. Submit the production run

```bash
cd /glade/work/kenzhao/saqqarleq-fjord-les
qsub -v CASE=control submit_casper.sh
qstat -u kenzhao
```

One A100 job, 10 model days or a clean stop ~30 min before the 24 h wall (whichever first).
`submit_casper.sh` already sets `CUDA_VISIBLE_DEVICES=0` and skips the cuda module.

**First leg: run it short.** Before committing the full 10 days, launch with `--stop_days` small
(edit the flag in `submit_casper.sh`, or run one interactively) and plot the plume with
`plot_quicklook.py` to confirm the outflow is sitting at the right depth on the bathymetry. Then
resubmit for the full run.

## 5. Monitor

```bash
tail -f logs/control.out
```

Healthy signs: CFL staying < 0.5, the CG Poisson solver converging each step, no NaNs, `max|u|`
around 0.2–0.3 m/s. Outputs land in `output/control/` (`control_face*.nc`, `control_mooring.nc`,
`control_xsect.nc`, `control_timeavg.nc`) plus `checkpoint_control_*`.

## 6. If it hits the wall before 10 days, resume

```bash
qsub -v CASE=control submit_casper.sh
```

Same command: it finds the checkpoint in `output/control/` and picks up automatically.

## 7. Pull results back to your laptop

```bash
mkdir -p ~/Desktop/Sanchez26Sarqar/cg_runs/control
rsync -avh --exclude 'checkpoint_*' \
  'kenzhao@data-access.ucar.edu:/glade/work/kenzhao/saqqarleq-fjord-les/output/control/*.nc' \
  ~/Desktop/Sanchez26Sarqar/cg_runs/control/
```

Then compare against the original `outer` run to confirm the CG solver removed the spurious
near-bathymetry tracer signal.
