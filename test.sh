#!/bin/bash
#
#-------------------------------------------------------------------------------
# setup.sh
#
# Usage:
#  $ sh test.sh 
#
#
#-------------------------------------------------------------------------------

set -u
set -e

unset TMPFILE

atexit() {
    [[ -n "${TMPFILE-}" ]] && rm -f "$TMPFILE"
}

# To trap an error caused by "set -e"
onerror()
{
    status=$?
    script=$0
    line=$1
    shift

    args=
    for i in "$@"; do
        args+="\"$i\" "
    done

    echo ""
    echo "------------------------------------------------------------"
    echo "Error occured on $script [Line $line]: Status $status"
    echo ""
    echo "PID: $$"
    echo "User: $USER"
    echo "Current directory: $PWD"
    echo "Command line: $script $args"
    echo "------------------------------------------------------------"
    echo ""
}

trap atexit EXIT
trap 'trap - EXIT; atexit; exit -1' INT PIPE TERM
trap 'onerror $LINENO "$@"' ERR


TMPFILE=$(mktemp "/tmp/${0##*/}.tmp.XXXXXXX")
echo TMPFILE=${TMPFILE}


function echoRed() {
    printf "\033[0;31m"
    echo "$*"
    printf "\033[0;39m"
}

function echoGreen() {
    printf "\033[0;32m"
    echo "$*"
    printf "\033[0;39m"
}

function echoCyan() {
    printf "\033[0;36m"
    echo "$*"
    printf "\033[0;39m"
}

function get_script_dir() {
    pushd `dirname $0` >/dev/null
    DIR=`pwd`
    popd >/dev/null
    echo $DIR
}

echo --------------------------------------------------------------------
#------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------
TMP_DIR=/tmp/tapas-build
SRC_ROOT=`get_script_dir`

if [[ -z "${SCALE-}" ]]; then
    SCALE=s
fi

STATUS=0
MAX_ERR="5e-2"

mkdir -p $TMP_DIR
cd $TMP_DIR

echo test.sh
echo Started at $(date)
echo "hostname=" $(hostname)
echo "compiler = ${COMPILER}"
echo "scale = ${SCALE}"

echo date
date

echo pwd
pwd

echo "SRC_ROOT=$SRC_ROOT"
echo "TMP_DIR=$TMP_DIR"

echo "ls /usr/bin/gcc*"
ls /usr/bin/gcc*
echo "ls /usr/bin/g++*"
ls /usr/bin/g++*

echo PATH=$PATH

export CXX=$COMPILER
export CC=$(echo $CXX | sed -e 's/clang++/clang/' | sed -e 's/g++/gcc/' | sed -e 's/icpc/icc/')

echo CC=$(which ${CC})
echo CXX=$(which ${CXX})

# detect MPI implementation
if [[ ! -x "${MPICXX:-}" ]]; then
    if mpicxx --showme:version 2>/dev/null | grep "Open MPI"; then
        # Opne MPI
        #MPICC="env OMPI_CC=${CC} mpicc"
        MPICXX="env OMPI_CXX=${CXX} mpicxx"

        if [[ -z "${MPIEXEC:-}" ]]; then
            MPIEXEC=mpiexec
        fi
        
        echo Looks like Open MPI.
    else
        # mpich family (mpich and mvapich)
        #MPICC="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicc"
        MPICXX="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicxx"
        
        if [[ -z "${MPIEXEC:-}" ]]; then
            MPIEXEC=mpiexec
        fi
        
        echo Looks like Mpich.
    fi
fi

echo Checking if compiler works
echo ${CXX} --version
${CXX} --version || {
    echoRed "ERROR: Compiler '${CXX}' seems to be broken"
    exit 1
}

echo Detecting mpicxx implementation

echo MPICXX=${MPICXX}
#echo MPICC=${MPICC}

echo ${MPICXX} -show
${MPICXX} -show

echo $CXX --version
$CXX --version

if [[ -d "${MYTH_DIR:-}" ]]; then
    echo MassiveThreads is activated. MYTH_DIR=${MYTH_DIR}
else
    echo MassiveThreads is NOT activated.
fi    

function test_unit() {
    echo --------------------------------------------------------------------
    echo C++ Unit Tests
    echo --------------------------------------------------------------------
    
    SRC_DIR=$SRC_ROOT/cpp/tests

    echoCyan make MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $SRC_DIR clean
    make -j MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $SRC_DIR clean

    echoCyan make MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $SRC_DIR all
    make -j MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $SRC_DIR all &&:
    echo $?

    TEST_TARGETS=$(make -C $SRC_DIR list | grep -v make | grep -v echo | grep test_)
    for t in $TEST_TARGETS; do
        for NP in 1 2 3 4; do
            echoCyan ${MPIEXEC} -np $NP $SRC_DIR/$t
            ${MPIEXEC} -np $NP $SRC_DIR/$t
        done
    done
}

test_unit

# Call BH test
/bin/bash $SRC_ROOT/sample/barnes-hut/test.sh

# Call FMM test
/bin/bash $SRC_ROOT/sample/exafmm-dev-13274dd4ac68/test.sh

# Check some special cases

# echo
# echo --------------------------------------------------------------------
# echo "ExaFMM (with Ncrit > Nbodies)"
# echo --------------------------------------------------------------------
# echo

# for MUTUAL in "" "_mutual" ; do
#     rm -f $TMPFILE; sleep 0.3s
#     echoCyan ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 500 -c 1024 -d c
#     ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 500 -c 1024 -d c  > $TMPFILE
#     if [[ ! "${QUIET:-}" == "1" ]]; then
#         cat $TMPFILE ||:
#     fi
    
#     accuracyCheck $TMPFILE
# done

# echo
# echo --------------------------------------------------------------------
# echo "ExaFMM (with Ncrit = 1)"
# echo --------------------------------------------------------------------
# echo
# for MUTUAL in "" "_mutual" ; do
#     rm -f $TMPFILE; sleep 0.3s
#     echoCyan ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 200 -c 1 -d c
#     ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 200 -c 1 -d c  > $TMPFILE
#     if [[ ! "${QUIET:-}" == "1" ]]; then
#         cat $TMPFILE ||:
#     fi
    
#     accuracyCheck $TMPFILE
# done

# if [[ $STATUS -eq 0 ]]; then
#     echo OK.
# else
#     echoRed "***** Test failed."
# fi

exit $STATUS

# Check the GPU version if nvcc is available
if which nvcc >/dev/null 2>&1; then
    NVCC_OPT="-O3 -Xcicc -Xptas --compiler-options -Wall --compiler-options -Wextra --compiler-options -Wno-unused-parameter \
            -Xcompiler -rdynamic -lineinfo --device-debug -x cu -arch sm_35 -ccbin=g++"

    which g++
    BIN=parallel_tapas_cuda
    SRC_DIR=$SRC_ROOT/sample/exafmm-dev-13274dd4ac68/examples
    compile=$($MPICXX -show -cxx=nvcc -DTAPAS_DEBUG=0 -DUSE_MPI -g $NVCC_OPT \
                      -DASSERT -DTAPAS_USE_VECTORMAP -DFP64 -DSpherical -DEXPANSION=6 -DTAPAS_LOG_LEVEL=0 \
                      -I$SRC_DIR/../include -I$SRC_DIR -I$SRC_DIR/../../../cpp/include \
                      -std=c++11 $SRC_DIR/../kernels/LaplaceP2PCPU.cxx $SRC_DIR/../kernels/LaplaceSphericalCPU.cxx \
                      $SRC_DIR/tapas_exafmm.cxx -o $BIN)

    # nvcc doesn't spport -Wl,... options
    compile=$(echo $compile | sed -e 's/-Wl,[^ ]*//g')
    echo $compile
    $compile

    for dist in ${DIST[@]}; do
        for nb in ${NB_FMM[@]}; do
            for ncrit in ${NCRIT[@]}; do
                echoCyan ${MPIEXEC} -n 1 ./$BIN --numBodies 1000
                ${MPIEXEC} -n 1 ./$BIN --numBodies 10000 > $TMPFILE
                if [[ ! "${QUIET:-}" == "1" ]]; then
                    cat $TMPFILE
                fi
                accuracyCheck $TMPFILE
            done
        done
    done

fi
