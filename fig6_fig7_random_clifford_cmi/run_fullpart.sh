#!/bin/bash
module load anaconda/3/2023.03 2>/dev/null
cd ~/fullpart
for combo in "4 0.01" "6 0.01" "8 0.01" "4 0.02" "6 0.02" "8 0.02" "4 0.05" "6 0.05" "8 0.05"; do echo "$combo"; done | OMP_NUM_THREADS=2 xargs -P 12 -L 1 sh -c 'python3 full_partition_cmi.py $0 $1 0 200 12 > fullpart_run_$0_$1.log 2>&1'
touch ALL_DONE
