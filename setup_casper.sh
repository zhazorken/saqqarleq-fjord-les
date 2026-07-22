#!/bin/bash -l
# setup_casper.sh — one-time environment setup on Casper (NCAR), run from the repo directory on a
# LOGIN node (needs network for Pkg). Resolves the Oceananigans 0.109 environment fresh (no
# Manifest is shipped).
#
#   cd /glade/work/$USER/saqqarleq-fjord-les
#   ./setup_casper.sh
set -e
cd "$(dirname "$0")"

# No cuda module: CUDA.jl ships its own toolkit and talks to the node driver directly. Loading the
# system cuda module can shadow that and cause version mismatches.
module purge
module load ncarenv/23.10
module list

# Use a juliaup Julia >= 1.10.11 (NOT the julia/1.10.5 module — its Pkg resolver writes a broken
# Manifest: KeyError "GPUArraysCore"). Ovall26 used 1.10.11. Install once:
#   curl -fsSL https://install.julialang.org | sh -s -- --yes && source ~/.bashrc && juliaup add 1.10
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
command -v "$JULIA" >/dev/null 2>&1 || { echo "ERROR: '$JULIA' not found — install juliaup (see comment above)."; exit 1; }
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
echo "Julia: $($JULIA --version)   depot: $JULIA_DEPOT_PATH"

# Resolve + precompile (adds Oceananigans 0.109, NCDatasets, Interpolations, etc. to the depot).
# On Julia >= 1.10.11 this is clean. If you ever see KeyError("GPUArraysCore"), you're on the
# 1.10.5 module — switch to juliaup 1.10 and `rm -f Manifest.toml` before re-running.
$JULIA --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo
echo "Environment ready. Quick GPU sanity check on a compute node (a couple of minutes):"
echo "    qsub -I -A UGIT0046 -q casper -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=a100 -l walltime=00:30:00"
echo "    # then on the node:"
echo "    module purge; module load ncarenv/23.10"
echo "    export JULIA_DEPOT_PATH=/glade/work/\$USER/.julia"
echo "    export CUDA_VISIBLE_DEVICES=0   # Casper hands it out as a GPU UUID this CUDA.jl can't parse"
echo "    $JULIA --project iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01"
echo "Then submit the real run:  qsub -v CASE=control submit_casper.sh"
