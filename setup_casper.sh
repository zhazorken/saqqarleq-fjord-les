#!/bin/bash -l
# setup_casper.sh — one-time environment setup on Casper (NCAR), run from the repo directory.
# Resolves + precompiles the Oceananigans 0.109 / Julia 1.10 environment for the GPU runs.
#
#   git clone <your-repo-url> saqqarleq-fjord-les && cd saqqarleq-fjord-les
#   JULIA=/path/to/julia ./setup_casper.sh
#   qsub -v CASE=control submit_casper.sh
set -e
cd "$(dirname "$0")"

# Instantiating packages needs NO HPC modules (CUDA.jl bundles its own toolkit; the GPU driver is
# only needed at run time). If `which julia` is empty, install juliaup (Julia >= 1.10) first:
#   curl -fsSL https://install.julialang.org | sh -s -- --yes && source ~/.bashrc
JULIA="${JULIA:-julia}"
command -v "$JULIA" >/dev/null 2>&1 || { echo "ERROR: '$JULIA' not found — install Julia >= 1.10 first."; exit 1; }
echo "Julia: $($JULIA --version)   depot: ${JULIA_DEPOT_PATH:-$HOME/.julia}"

# No Manifest is shipped, so this resolves Oceananigans 0.109.x fresh, then precompiles.
$JULIA --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo
echo "Environment ready. Quick GPU sanity check (a couple of minutes):"
echo "    $JULIA --project --pkgimages=no iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01"
echo "Then submit the production runs:"
echo "    qsub -v CASE=control submit_casper.sh"
echo "    qsub -v CASE=tide    submit_casper.sh"
echo "    qsub -v CASE=pump    submit_casper.sh"
