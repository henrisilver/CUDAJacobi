#!/bin/bash

gcc test/input_generator.c -o test/input_generator
#cd stats/
#gcc metrics.c -o metrics-analyzer
#cd ..
gcc sequential/sequential.c -o sequential/sequential
nvcc parallel/parallel.cu -o parallel/parallel
