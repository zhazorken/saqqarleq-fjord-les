# Running a scenario on Casper (NCAR)

Close to your `outerpump3`/`outertide3` runs (ncarenv 23.10 + cuda + peak-memusage, 40 GB, 24 h),
but on an A100 with a juliaup Julia >= 1.10.11 and the `ConjugateGradientPoissonSolver` in
`iceplume.jl` (`cg_reltol` 1e-4, ~30 m horizontal grid). Below runs the **control** case; swap
`CASE=control` for `tide`, `pump`, or `tidepump`.

## 0. Allocation

`submit_casper.sh` and the interactive command below use `-A UGIT0046` (your current plume/DNS
allocation). Everything else matches the original runs.

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
module purge; module load ncarenv/23.10 cuda
export JULIA_DEPOT_PATH=/glade/work/kenzhao/.julia
~/.juliaup/bin/julia --project iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01
```

Watch for: "Model built", the CG solver converging, `max|u|` growing (not NaN). Then `exit`.

## 4. Submit the production run

```bash
cd /glade/work/kenzhao/saqqarleq-fjord-les
qsub -v CASE=control submit_casper.sh
qstat -u kenzhao
```

One A100 job, 10 model days or a clean stop ~30 min before the 24 h wall (whichever first).

## 5. Monitor

```bash
tail -f logs/control.out
```

Healthy signs: CFL staying < 0.5, the CG Poisson solver converging each step, no NaNs. Outputs
land in `output/control/` (`control_face*.nc`, `control_mooring.nc`, `control_xsect.nc`,
`control_timeavg.nc`) plus `checkpoint_control_*`.

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
