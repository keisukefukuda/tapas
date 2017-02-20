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
echo "compiler=" ${CXX}
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

if [[ -z "${CXX:-}" ]]; then
    CXX=g++
fi

echo CXX=$(which ${CXX})


if [[ -d "${MYTH_DIR:-}" ]]; then
    echo MassiveThreads is activated. MYTH_DIR=${MYTH_DIR}
else
    echo MassiveThreads is NOT activated.
fi    

# Call unit test
/bin/bash $SRC_ROOT/tests/test.sh

# Call BH test
/bin/bash $SRC_ROOT/sample/barnes-hut/test.sh

# Call FMM test
/bin/bash $SRC_ROOT/sample/fmm/test.sh

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
