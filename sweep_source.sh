#!/bin/bash
# sweep_source.sh — launch the source-sensitivity sweep on Casper (run on a LOGIN node).
#
# Fires one short PUMP run per config via sweep_worker.sh. Each config concentrates the plume source
# a bit more than the last, so you can see whether the mooring M2 velocity swing climbs back toward
# the old ~0.025 and whether it stays numerically stable:
#
#   s0_baseline   current diffuse source (reproduces ~0.01)         [reference]
#   s1_thin       --z_src_bot=-25                                   (thin the outflow layer)
#   s2_thinstiff  --z_src_bot=-25 --sig_src=25                      (thin + stiffer relaxation)
#   s3_narrow     --z_src_bot=-25 --sig_src=25 --y_src=90           (+ narrower in y)
#   s4_gain       ... --src_gain=1.4                                (+ stronger target; needs patched iceplume.jl)
#
# Usage (from the repo dir, on a login node):
#   ./sweep_source.sh                       # all five configs, DAYS=3 each
#   ./sweep_source.sh s0_baseline s2_thinstiff s4_gain   # a cheaper subset
#   DAYS=2 ./sweep_source.sh                 # shorter (cheaper) runs
#
# NOTE: s4_gain uses --src_gain, which only exists in the patched iceplume.jl. If you did not sync
# that edit to Casper, run the first four (or drop s4_gain from the list) and it still answers the
# core question. Smoke-test the patched model once before submitting:
#   $HOME/.juliaup/bin/julia --project iceplume.jl --arch=cpu --simname=cputest --stop_days=0.02
set -euo pipefail

DAYS="${DAYS:-3}"
ALL=(s0_baseline s1_thin s2_thinstiff s3_narrow s4_gain)
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${ALL[@]}")

[ -f sweep_worker.sh ] || { echo "run this from the repo dir that has sweep_worker.sh"; exit 1; }

echo "Sweep: DAYS=$DAYS   configs: ${CONFIGS[*]}"
for tag in "${CONFIGS[@]}"; do
  jid=$(qsub -N "sw_$tag" -v "TAG=$tag,DAYS=$DAYS" sweep_worker.sh)
  printf '  %-13s -> %s   (output/sweep/%s/%s_mooring.nc)\n' "$tag" "$jid" "$tag" "$tag"
done

echo
echo "Monitor:  qstat -u \$USER"
echo "When they finish, measure with measure_source_sweep.py (see its header), e.g.:"
echo "  module load conda; conda activate npl"
echo "  python measure_source_sweep.py --runs output/sweep --out figures/sweep"
