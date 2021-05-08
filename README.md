# GPU Mathy

This repository consists of codes for the Mathy Compiler. This compiler is a part of the CS6023 Course Project.

The Mathy compiler generates ready-to-run cuda code for commonly used mathematical expressions like forall, summation, product etc.

The project was written in C++ using Flex and Bison.

## Installing Dependencies
- g++\
    `sudo apt install g++`
- flex\
    `sudo apt install flex`
- bison\
    `sudo apt install bison`

## Compiling Project
Run `make` in the `src` directory to generate the executable `a.out`
This will generate some intermediate files also.

## Converting Mathy to Cuda
If the Mathy code is present in input.txt, running `./a.out input.txt output.cu` will create a file `output.cu` which will contain the equivalent Cuda code. The program assumes that all arrays are of type float.

Run `make clean` to remove the executable and intermediate files.

## Benchmarks
Run `python3 benchmark.py` to generate the cuda files for mathy text files present in [benchmark/input](https://github.com/dl-thops/gpu-mathy/tree/main/benchmark/input) and write the outputs to [benchmark/output](https://github.com/dl-thops/gpu-mathy/tree/main/benchmark/output).
The benchmarks have been implemented based on [PolyBench benchmarks](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/). 
The results of the output code against the benchmark code have been agglomerated in this [Report](https://github.com/dl-thops/gpu-mathy/tree/main/Report.pdf).

Please refer to the [wiki](https://github.com/dl-thops/gpu-mathy/wiki#welcome-to-the-gpu-mathy-wiki) for the Grammar and features of the Mathy language.

This project was done under the guidance of Prof. [Rupesh Nasre](https://www.cse.iitm.ac.in/~rupesh/)
