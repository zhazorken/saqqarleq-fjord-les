#!/bin/bash -l
# setup_casper.sh — one-time environment setup on Casper (NCAR), run from the repo directory on a
# LOGIN node (needs network for Pkg). Uses the same modules as your production runs and resolves
# the Oceananigans 0.109 environment fresh (no Manifest is shipped).
#
#   cd /glade/work/$USER/saqqarleq-fjord-les
#   ./setup_casper.sh
set -e
cd "$(dirname "$0")"

module purge
module load ncarenv/23.10
module load julia/1.10.5 cuda
module list

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
echo "Julia: $(julia --version)   depot: $JULIA_DEPOT_PATH"

# Resolve + precompile (adds Oceananigans 0.109, NCDatasets, Interpolations, etc. to the depot).
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo
echo "Environment ready. Quick GPU sanity check on a compute node (a couple of minutes):"
echo "    qsub -I -A UGIT0046 -q casper -l select=1:ncpus=1:mem=40GB:ngpus=1 -l gpu_type=v100 -l walltime=00:30:00"
echo "    # then on the node:"
echo "    module load ncarenv/23.10 julia/1.10.5 cuda"
echo "    export JULIA_DEPOT_PATH=/glade/work/\$USER/.julia"
echo "    julia --project iceplume.jl --arch=gpu --simname=gpucheck --stop_days=0.01"
echo "Then submit the real run:  qsub -v CASE=control submit_casper.sh"
