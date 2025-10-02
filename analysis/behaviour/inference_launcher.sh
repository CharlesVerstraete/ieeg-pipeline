#!/bin/bash

# subjects=(2 3 4 5 8 9 12 14 16 19 20 23 25 28)
# n_cores=104
# n_subjects=${#subjects[@]}
# cores_per_subject=$((n_cores / n_subjects))
# start_core=8

# for subj in "${subjects[@]}"; do
#     end_core=$((start_core + cores_per_subject - 1))
#     export SUBJECT="$subj"
#     export N_SAMPLES="1500"  
#     export N_CHAINS="$cores_per_subject"
#     log_file="/home/cverstraete/nasShare/projects/cverstraete/log_subj${subj}.txt"
#     echo "Launching subject $subj on cores $start_core-$end_core"
#     nohup taskset -c ${start_core}-${end_core} julia --threads $cores_per_subject /home/cverstraete/ieeg_stratinf_analysis/v2/behaviour/inference.jl > "${log_file}" 2>&1 &
#     start_core=$((end_core + 1))
# done

# wait
# echo "Tous les jobs sont lancés."




n_cores=8
n_subjects=${#subjects[@]}
cores_per_subject=8  #$((n_cores / n_subjects))
start_core=0

# for subj in "${subjects[@]}"; do
subj=28
end_core=$((start_core + cores_per_subject - 1))
export SUBJECT="$subj"
export N_SAMPLES="1500"  
export N_CHAINS=8  #"$cores_per_subject"
log_file="/home/cverstraete/nasShare/projects/cverstraete/log_subj${subj}.txt"
echo "Launching subject $subj on cores $start_core-$end_core"
nohup taskset -c ${start_core}-${end_core} julia --threads $cores_per_subject /home/cverstraete/ieeg_stratinf_analysis/v2/behaviour/inference.jl > "${log_file}" 2>&1 &
start_core=$((end_core + 1))
# done

wait
echo "Tous les jobs sont lancés."