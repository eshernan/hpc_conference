#!/bin/bash 
# for serial compilation
gcc -o matmul_serial matmul_serial.c

# For parallel compilation
gcc -fopenmp -o matmul_parallel matmul_parallel.c

echo "Compile Finished ....." 