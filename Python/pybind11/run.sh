#!/bin/bash


python benchmark_map.py -type dict -N 191000000 -ingestion_batch_size 1000000 -lookup_batch_size 10000
python benchmark_map.py -type map -N 191000000 -ingestion_batch_size 1000000 -lookup_batch_size 10000
