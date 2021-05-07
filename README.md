# gpu-mathy

## This repository consists of codes for the Mathy Compiler. This compiler is a part of the CS6023 Course Project.
### The Mathy compiler generates ready-to-run cuda code for commonly used mathematical expressions like forall,summation,product etc.

### Installing Dependencies
g++ 
`sudo apt install g++`
python3 (Optional, For running benchmarks)
`sudo apt install python3`
flex
`sudo apt install flex`
bison
`sudo apt install bison`

#### Generating executables
run `make` to generate `./a.out`
This will generate some intermediate files also.

#### Converting Mathy to Cuda
If the Mathy code is present in input.txt,
running `./a.out input.txt output.cu` will create a file output.cu which will contain the equivalent Cuda code.
The program assumes that all arrays are of type float.

Run `make clean` to remove the executable and intermediate files.

