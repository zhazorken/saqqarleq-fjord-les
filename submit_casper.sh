#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N saqq_cg
#PBS -k eod
#PBS -o logs/saqq_cg.out
#PBS -e logs/saqq_cg.err
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=a100
#PBS -M kenzhao@unc.edu
#PBS -m abe
#
# Saqqarleq fjord LES on Casper. One script, iceplume.jl: closed domain, interior plume source at
# neutral buoyancy, ConjugateGradientPoissonSolver by default (the immersed-bottom tracer fix;
# validated at Δt ~ 15 s). Add EXTRA="--fft=1" for the FFT cross-check (same run, default solver).
#
#     qsub -v CASE=control  submit_casper.sh                    # steady discharge, no tide (default)
#     qsub -v CASE=tide     submit_casper.sh                    # + external M2 tide
#     qsub -v CASE=pump     submit_casper.sh                    # tidally modulated discharge
#     qsub -v CASE=control,EXTRA=--fft=1 submit_casper.sh       # FFT solver instead of CG
#
# Julia: a juliaup >= 1.10.11 (NOT the julia/1.10.5 module — its Pkg resolver writes a broken
# Manifest). No cuda module: CUDA.jl bundles its own toolkit and uses the node driver. And we pin
# CUDA_VISIBLE_DEVICES=0 because Casper hands it out as a GPU UUID that this CUDA.jl doesn't parse.

cd "$PBS_O_WORKDIR" || exit 1
mkdir -p logs

module purge
module load ncarenv/23.10
module load peak-memusage
module list

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
export CUDA_VISIBLE_DEVICES=0
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
SCRIPT="${SCRIPT:-iceplume.jl}"
EXTRA="${EXTRA:-}"   # e.g. EXTRA="--fft=1" to use the FFT solver instead of CG

CASE=${CASE:-control}
case "$CASE" in
  control)  FLAGS="--tide=0 --pump=0" ;;
  tide)     FLAGS="--tide=1 --pump=0" ;;
  pump)     FLAGS="--tide=0 --pump=1" ;;
  tidepump) FLAGS="--tide=1 --pump=1" ;;
  *) echo "unknown CASE='$CASE' (control|tide|pump|tidepump)"; exit 1 ;;
esac

# Outputs + checkpoints per case (off the git repo; auto-resumes if resubmitted with the same CASE).
OUTDIR="${OUTDIR:-$PBS_O_WORKDIR/output/$CASE}"
mkdir -p "$OUTDIR"

# --wall_time_limit stops cleanly ~30 min before the 24 h PBS wall so the checkpoint is complete.
peak_memusage $JULIA --project "$SCRIPT" \
    --simname="$CASE" $FLAGS $EXTRA --arch=gpu --outdir="$OUTDIR" \
    --stop_days=10 --wall_time_limit=23.5 \
    2>&1 | tee logs/${CASE}.out
