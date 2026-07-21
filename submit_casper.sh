#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N saqq_cg
#PBS -k eod
#PBS -o logs/saqq_cg.out
#PBS -e logs/saqq_cg.err
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=v100
#PBS -M kenzhao@unc.edu
#PBS -m abe
#
# Same environment as your outerpump3/outertide3 runs (ncarenv 23.10 + julia/1.10.5 + cuda +
# peak-memusage, v100, 40 GB, 24 h). The ONLY change from the originals is that iceplume.jl now
# uses the ConjugateGradientPoissonSolver. Pick the scenario with -v CASE=:
#
#     qsub -v CASE=control  submit_casper.sh     # constant discharge, no tide   (default)
#     qsub -v CASE=tide     submit_casper.sh     # constant discharge + M2 tide
#     qsub -v CASE=pump     submit_casper.sh     # tidally modulated discharge
#     qsub -v CASE=tidepump submit_casper.sh     # both
#
# Account is UGIT0046 (your current plume/DNS allocation). Everything else matches the originals.

cd "$PBS_O_WORKDIR" || exit 1
mkdir -p logs

# Exactly the module stack your Dec-2024 run used (proven to work on Casper).
# NOTE: use a juliaup Julia >= 1.10.11, NOT the julia/1.10.5 module (its Pkg resolver writes a
# broken Manifest — KeyError "GPUArraysCore"). Ovall26 used 1.10.11 for the same reason.
module purge
module load ncarenv/23.10
module load cuda
module load peak-memusage
module list

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"

CASE=${CASE:-control}
case "$CASE" in
  control)  FLAGS="--tide=0 --pump=0" ;;
  tide)     FLAGS="--tide=1 --pump=0" ;;
  pump)     FLAGS="--tide=0 --pump=1" ;;
  tidepump) FLAGS="--tide=1 --pump=1" ;;
  *) echo "unknown CASE='$CASE' (control|tide|pump|tidepump)"; exit 1 ;;
esac

# Outputs + checkpoints for this case (kept out of the git repo; auto-resumes if resubmitted).
OUTDIR="${OUTDIR:-$PBS_O_WORKDIR/output/$CASE}"
mkdir -p "$OUTDIR"

# --wall_time_limit stops cleanly ~30 min before the 24 h PBS wall so the checkpoint is complete;
# resubmitting the same CASE then picks up from it. --stop_days=10 matches the original run length.
peak_memusage $JULIA --project iceplume.jl \
    --simname="$CASE" $FLAGS --arch=gpu --outdir="$OUTDIR" \
    --stop_days=10 --wall_time_limit=23.5 \
    2>&1 | tee logs/${CASE}.out
