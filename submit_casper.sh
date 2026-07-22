#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N saqq_cg
#PBS -j oe
#PBS -r n
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=a100
#PBS -M kenzhao@unc.edu
#PBS -m abe
#
# PBS writes its own log to saqq_cg.o<jobid> in the submission directory (no pre-existing subdir
# needed — an -o into a missing logs/ dir aborts the job at launch and, if rerunnable, requeues it
# forever). -r n means a failure stops as status F so you can read the error instead of looping.
# The human-readable per-case log is logs/<case>.out (created by this script, below).
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
# Manifest). No cuda module: CUDA.jl bundles its own toolkit and uses the node driver. Casper hands
# out the GPU as a UUID this CUDA.jl can't parse, so below we translate it to the right numeric index
# (see the note there — hardcoding 0 fails on shared nodes when your GPU isn't index 0).

cd "$PBS_O_WORKDIR" || exit 1
mkdir -p logs

module purge
module load ncarenv/23.10
module load peak-memusage
module list

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"

# --- GPU selection + node health check ----------------------------------------------------------
# Casper assigns the GPU via CUDA_VISIBLE_DEVICES=<UUID>, which this CUDA.jl can't parse. The right
# fix depends on how many GPUs are actually visible in the job's cgroup:
#   exactly one visible -> it is local index 0, use 0
#   several visible     -> translate the assigned UUID to its numeric index
# We also print nvidia-smi so a bad "black-hole" node (GPU missing) is obvious in the PBS log, and we
# refuse to launch Julia on a node with no usable GPU (so it fails fast and clearly, not cryptically).
echo "host=$(hostname)  PBS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "--- nvidia-smi -L ---"; nvidia-smi -L || echo "nvidia-smi: no GPU / driver error on $(hostname)"
nvis=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
if [[ "$nvis" -eq 0 ]]; then
    echo "FATAL: no usable GPU on $(hostname) — likely a bad node. Report it and resubmit." >&2
    exit 1
elif [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    if [[ "$nvis" -le 1 ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        gpu_idx=$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader \
                  | awk -F', *' -v u="$CUDA_VISIBLE_DEVICES" '$1==u{print $2; exit}')
        export CUDA_VISIBLE_DEVICES="${gpu_idx:-0}"
    fi
fi
echo "using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES on $(hostname)"
# ------------------------------------------------------------------------------------------------

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
