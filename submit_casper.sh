#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N saqq_fjord
#PBS -o logs/saqq_fjord.log
#PBS -j oe
#PBS -l walltime=11:59:00
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=60gb:ngpus=1:gpu_type=a100
#PBS -M kenzhao@unc.edu
#PBS -m ae
#PBS -r n
#
# One job = one scenario. Pick with -v CASE=:
#     qsub -v CASE=control submit_casper.sh     # constant discharge, no tide  (default)
#     qsub -v CASE=tide    submit_casper.sh     # constant discharge + external M2 tide
#     qsub -v CASE=pump    submit_casper.sh     # tidally modulated discharge (Varying-SGD)
#     qsub -v CASE=tidepump submit_casper.sh    # both together
# Override Julia / output dir:  -v CASE=pump,JULIA=/path/to/julia,OUTDIR=/glade/derecho/scratch/$USER/saqq
#
# ~25 M cells (522×604×80). No HPC modules needed: CUDA.jl bundles its toolkit and uses the GPU
# node's driver. Do NOT force-purge/load a fixed ncarenv (pulls a broken openmpi on Casper).
# `--pkgimages=no` avoids a known cluster precompile issue.

cd "$PBS_O_WORKDIR" || exit 1
mkdir -p logs

JULIA="${JULIA:-julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; Julia: $($JULIA --version 2>/dev/null)"

CASE=${CASE:-control}
OUTDIR="${OUTDIR:-/glade/work/$USER/saqq_fjord_runs/$CASE}"
mkdir -p "$OUTDIR"

case "$CASE" in
  control)  FLAGS="--tide=0 --pump=0" ;;
  tide)     FLAGS="--tide=1 --pump=0" ;;
  pump)     FLAGS="--tide=0 --pump=1" ;;
  tidepump) FLAGS="--tide=1 --pump=1" ;;
  *) echo "unknown CASE='$CASE' (control|tide|pump|tidepump)"; exit 1 ;;
esac

# Auto-resumes from a checkpoint in OUTDIR if the same job is re-submitted with the same CASE.
time $JULIA --project --pkgimages=no iceplume.jl \
    --simname="$CASE" $FLAGS --outdir="$OUTDIR" \
    --stop_days=10 --wall_time_limit=11.5 \
    2>&1 | tee logs/${CASE}.out

qstat -f $PBS_JOBID >> logs/saqq_fjord.log
