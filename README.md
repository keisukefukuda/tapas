Tapas - Parallel Framework for Tree-based Adaptively PArtitioned Space

[![Build Status](https://travis-ci.org/keisukefukuda/tapas.svg?branch=master)](https://travis-ci.org/keisukefukuda/tapas)

This software is released under the MIT License, see LICENSE.txt

## How to build Tapas examples

### Compiler Requirement

Tapas requires C++11 compiler. Recommended compiler versions are:

|Compiler               | Version |
|:----------------------|:--------|
|GNU C++ compiler       | >= 4.9  |
|LLVM clang++           | >= 3.6  |
|Intel C++ compiler     | >= 2015 |


### The FMM example 

Basic compilation:

    $ cd sample/exafmm-dev-13274dd4ac68/examples
    $ mpicxx -std=c++11 -O2 -lrt tapas_exafmm.cxx -I../include -I../../../cpp/include -DUSE_MPI -DSpherical -DEXPANSION=10 -DFP64 -o parallel_tapas

* `USE_MPI` definition is for Tapas. MPI is not necessary for Tapas, but the current implementation does not support compiling without MPI and `USE_MPI` macro.
* `Spherical`, `EXPANSION`, and `FP64` are definitions for TapasFMM, which is [ExaFMM](https://github.com/exafmm/exafmm) ported to Tapas. 
   * Although the original ExaFMM supports `Spherical` and `Cartesian` kernels, but only `Spherical` is ported as of now. `EXPANSION` is degree of multipole/local expansion (typically 10). `FP64` is to use double precision.
   
To build multithreaded code using MassiveThreads, add `-DMTHREAD=1` and specify include/library path to your MassiveThreads installation.

    $ MYTH_DIR=YOUR-MYTH-DIR
    $ mpicxx (snip) -DMTHREAD=1 -I${MYTH_DIR}/include -L${MYTH_DIR}/lib -lmyth-native

Depending on how your MPI library is built, the mpicxx compiler may use its default compiler which does not support C++11.
In such a case, you can speicfy the underlying C++ compiler via environment variables. See the documentation of the MPI for details.

    $ # For mpich family (mpcih, mvapich, Intel MPI, etc.)
    $ export MPICH_CXX="your new C++ compiler"
    
For advanced optimization with Intel Compiler,

    $ mpicxx -std=c++11 -O2 -lrt tapas_exafmm.cxx -I../include -I../../../cpp/include -DUSE_MPI -DSpherical -DEXPANSION=10 -DFP64 -o parallel_tapas \
        -funroll-loops -xHOST -O3 -no-prec-div -fp-model fast=2 -no-inline-max-per-routine -no-inline-max-per-compile 
        
You may also want to inactivate debugging assertions:

    $ mpicxx (snip) -DNDEBUG -DTAPAS_DEBUG=0
    
How to run:

    $ mpiexec -np 10 ./parallel_tapas --numBodies 100000 --ncrit 64 --theta 0.34
    
* --nuBodies: Number of total bodies of the simulation (not number of bodies per process).
* --ncrit : Ncrit parameter of FMM (each leaf has at most Ncrit bodies)
* --theta : Multipole acceptance criteria

To control number of threads, use `MYTH_WORKER_NUM` environment variable for MassiveThreads.
        
## Build FAQ

Q. I'm using new icpc/mpicxx but parallel_tapas doesn't compile. Why?

A. Check the version of g++. `mpicxx` uses g++ as a backend. If it's old, install a new gcc on your local environment (we recommend >= 4.9.3). By using `mpicxx -show` command, you can check what is the underlying C++ compiler is and how it is invoked. Also note that Intel C++ compiler uses GCC's standard library implementation directly. Thus if the system's default g++ is old, its STL is not C++11-ready. 
    
## Preprocessor symbols

|Name                   | Possible values  | Default value | Description                                               |
|:----------------------|:-----------------|:--------------|:----------------------------------------------------------|
|TAPAS_DEBUG            | unset, 0, or 1   | unset         | Enable verbose debug output. Minor performance slowdown   |
|TAPAS_DEBUG_DUMP       | unset, 0, or 1   | unset         | Enable verbose debug output. Serious performance slowdown |
|TAPAS_REPORT_PREFIX    | filename prefix  | unset         | Prefix of performance report file names (*.csv)           |
|TAPAS_REPORT_SUFFIX    | part of filename | unset         | Suffix of performance report file names                   |
|TAPAS_DEBUG_COMM_MATRIX| unset/any        | unset         | Print Communication Matrix in MPI_Alltoallv()             |
|TAPAS_DEBUG_HISTOGRAM  | unset/any        | unset         | Print depth histogram of the tree                         |

