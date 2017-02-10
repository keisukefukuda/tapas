#!/bin/sh

#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

SCRIPT_DIR=$(dirname "${BASH_SOURCE:-$0}")
TAPAS_DIR=$(pushd ${SCRIPT_DIR}/../../ >/dev/null ; pwd -P; popd >/dev/null)

source $TAPAS_DIR/scripts/test_common.sh
source $TAPAS_DIR/sample/exafmm-dev-13274dd4ac68/test.sh load

function build_cuda() {
    local SRC_DIR=$FMM_DIR/examples
    pushd $SRC_DIR

    NVCC_OPT="-O3 -Xcicc -Xptas --compiler-options -Wall --compiler-options -Wextra --compiler-options -Wno-unused-parameter \
            -Xcompiler -rdynamic -lineinfo --device-debug -x cu -arch sm_35 -ccbin=g++"

    which g++
    BIN=parallel_tapas_cuda
    compile=$($MPICXX -show -cxx=nvcc -DTAPAS_DEBUG=0 -DUSE_MPI -g $NVCC_OPT \
                      -DASSERT -DTAPAS_USE_VECTORMAP -DFP64 -DSpherical -DEXPANSION=6 -DTAPAS_LOG_LEVEL=0 \
                      -I$SRC_DIR/../include -I$SRC_DIR -I$SRC_DIR/../../../cpp/include \
                      -std=c++11 $SRC_DIR/../kernels/LaplaceP2PCPU.cxx $SRC_DIR/../kernels/LaplaceSphericalCPU.cxx \
                      $SRC_DIR/tapas_exafmm.cxx -o $BIN)

    echo $compile
    
    popd
}


build_cuda

