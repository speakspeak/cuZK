#!/bin/bash

# this script is used to profile performance and energy consumption of MSM

# set up a log file
echo "Param,Mem,Graphics,Energy,Time" >> energy.log
# loop over arguments
for n in 20 21 22; do
    # loop over clock frequencies
    for mem in 877; do
        for freq in $(nvidia-smi -q -d SUPPORTED_CLOCKS | awk '/Graphics.*MHz/{ print $3 }'); do
            # change GPU clock
            sudo nvidia-smi -ac $mem,$freq
            for i in {1..3}; do
                # run profiled computations
                output=$(./msmtestb $n)
                # extract and report results
                time=$(echo $output | gawk 'match($0,/Time: ([0-9]+\.[0-9]+)/, a) {print a[1]}')
                energy=$(echo $output | gawk 'match($0,/Energy: ([0-9]+)/, a) {print a[1]}')
                result="Param=$n,Mem=$mem,Graphics=$freq,Energy=$energy,Time=$time"
                echo $result
                echo "$n,$mem,$freq,$energy,$time" >> energy.log
            done
        done
    done
done
