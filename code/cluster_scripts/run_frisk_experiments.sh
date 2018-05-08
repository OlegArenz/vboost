#!/bin/bash
for i in `seq 1 10`; do
    for j in `seq 1 5`; do
        python2.7 mlm_main.py -model frisk -rank $j -vboost -progressDir _output/rank${j}_seed${i}/ -seed $i
    done
done