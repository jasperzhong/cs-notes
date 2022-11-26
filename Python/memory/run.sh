#!/bin/bash

N=10000

for ((i=5; i<=8; i++)); do
    for ((j=1; j<=9; j++)); do
        M=$((j*$N))
        python test.py -N $M
    done
    N=$((N * 10))
done
