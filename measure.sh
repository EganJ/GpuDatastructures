#!/bin/bash
experiment=$2
build/main $1 > output1
mv metrics_output.json results/metrics_output_${experiment}_t1.json

for trial_num in $(seq 1 10); do
    build/main $1 > output1
    mv metrics_output.json results/metrics_output_${experiment}_t${trial_num}.json
done

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t2.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t3.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t4.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t5.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t6.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t7.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t8.json

# build/main $1 > output1
# mv metrics_output.json results/metrics_output_t9.json

