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
echo MPICXX=$(which mpicxx)

echo Checking if compiler works
echo ${CXX} --version
${CXX} --version || {
    echoRed "ERROR: Compiler '${CXX}' seems to be broken"
    exit 1
}

echo Detecting mpicxx implementation

# detect MPI implementation
if mpicxx --showme:version 2>/dev/null | grep "Open MPI"; then
    # Opne MPI
    MPICC="env OMPI_CC=${CC} mpicc"
    MPICXX="env OMPI_CXX=${CXX} mpicxx"

    if [[ -z "${MPIEXEC:-}" ]]; then
        MPIEXEC=mpiexec
    fi
    
    echo Looks like Open MPI.
else
    # mpich family (mpich and mvapich)
    MPICC="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicc"
    MPICXX="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicxx"
    
    if [[ -z "${MPIEXEC:-}" ]]; then
        MPIEXEC=mpiexec
    fi
    
    echo Looks like Mpich.
fi

echo MPICXX=${MPICXX}
echo MPICC=${MPICC}

echo mpicxx -show
mpicxx -show

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

    echoCyan make MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $SRC_DIR clean
    make -j MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $SRC_DIR clean

    echoCyan make MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $SRC_DIR all
    make -j MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $SRC_DIR all

    TEST_TARGETS=$(make -C $SRC_DIR list | grep -v make | grep -v echo | grep test_)
    for t in $TEST_TARGETS; do
        for NP in 1 2 3 4; do
            echoCyan ${MPIEXEC} -np $NP $SRC_DIR/$t
            ${MPIEXEC} -np $NP $SRC_DIR/$t
        done
    done
}

test_unit

function test_bh() {
    echo --------------------------------------------------------------------
    echo Barnes Hut
    echo --------------------------------------------------------------------

    MAX_ERR=1e-2

    if echo $SCALE | grep -Ei "^t(iny)?" >/dev/null ; then
        NP=(1)
        NB=(100)
    elif echo $SCALE | grep -Ei "^s(mall)?" >/dev/null ; then
        NP=(1 4)
        NB=(1000)
    elif echo $SCALE | grep -Ei "^m(edium)?" >/dev/null ; then
        NP=(1 2 3 4 5 6)
        NB=(1000 2000)
    elif echo $SCALE | grep -Ei "^l(arge)?" >/dev/null ; then
        NP=(1 2 4 8 16 32)
        NB=(1000 2000 4000 8000 16000)
    else
        echo "Unknown SCALE : '$SCALE'" >&2
        exit 1
    fi

    SRC_DIR=$SRC_ROOT/sample/barnes-hut
    BIN=$SRC_DIR/bh_mpi

    echoCyan make CXX=\"${CXX}\" CC=\"${CC}\" MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $SRC_DIR clean $(basename $BIN)
    make CXX=${CXX} CC=${CC} MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $SRC_DIR clean $(basename $BIN)

    for np in ${NP[@]}; do
        for nb in ${NB[@]}; do
            echoCyan ${MPIEXEC} -n $np $SRC_DIR/bh_mpi -w $nb
            ${MPIEXEC} -n $np $SRC_DIR/bh_mpi -w $nb | tee $TMPFILE

            PERR=$(grep "P ERR" $TMPFILE | grep -oE "[0-9.e+-]+|[+-]?nan")
            FERR=$(grep "F ERR" $TMPFILE | grep -oE "[0-9.e+-]+|[+-]?nan")

            if [[ $(python -c "print(float('$PERR') < $MAX_ERR)") == "False" ]]; then
                echoRed "*** Error check failed. P ERR $PERR > $MAX_ERR"
                STATUS=$(expr $STATUS + 1)
            else
                echoGreen P ERR OK
            fi
            if [[ $(python -c "print(float('$FERR') < $MAX_ERR)") == "False" ]]; then
                echoRed "*** Error check failed. F ERR $FERR > $MAX_ERR"
                STATUS=$(expr $STATUS + 1)
            else
                echoGreen F ERR OK
            fi
            echo
            echo
        done
    done
}

#test_bh

function build_fmm() {
    if echo $SCALE | grep -Ei "^t(iny)?" >/dev/null ; then
        NP=(1)
        NB=(100)
        DIST=(c)
        NCRIT=(1 2 16)
    elif echo $SCALE | grep -Ei "^s(mall)?" >/dev/null ; then
        NP=(1 2)
        NB=(1000)
        DIST=(c)
        NCRIT=(1 2 16)
    elif echo $SCALE | grep -Ei "^m(edium)?" >/dev/null ; then
        NP=(1 2 3 4 5 6)
        NB=(10000 20000)
        DIST=(s c)
        NCRIT=(1 2 16 64)
    elif echo $SCALE | grep -Ei "^l(arge)?" >/dev/null ; then
        NP=(1 2 4 8 16 32)
        NB=(10000 20000 40000 80000 160000)
        DIST=(l s p c)
        NCRIT=(1 2 16 64)
    else
        echo "Unknown SCALE : '$SCALE'" >&2
        exit 1
    fi

    SRC_DIR=$SRC_ROOT/sample/exafmm-dev-13274dd4ac68/examples

    make VERBOSE=1 -C $SRC_DIR clean 

    # Build multi-threaded version
    if [[ -d "${MYTH_DIR:-}" ]]; then
        export MYTH_DIR
        echoCyan env CC=${CC} CXX=${CXX} MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" make VERBOSE=1 MTHREAD=1 MODE=debug -C $SRC_DIR tapas
        env CC=${CC} CXX=${CXX} MPICC="${MPICC}" MPICXX="${MPICXX}" make VERBOSE=1 MTHREAD=1 MODE=debug -C $SRC_DIR tapas

        mv $SRC_DIR/parallel_tapas $SRC_DIR/parallel_tapas_mt
        mv $SRC_DIR/parallel_tapas_mutual $SRC_DIR/parallel_tapas_mutual_mt
    fi

    # Build single-threaded, without-weighted-repartitioning version
    echoCyan env CC=${CC} CXX=${CXX} MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" make VERBOSE=1 MODE=debug WEIGHT=0 -C $SRC_DIR tapas
    env CC=${CC} CXX=${CXX} MPICC="${MPICC}" MPICXX="${MPICXX}" make VERBOSE=1 MODE=debug WEIGHT=0 -C $SRC_DIR tapas

    mv $SRC_DIR/parallel_tapas $SRC_DIR/parallel_tapas_nw
    mv $SRC_DIR/parallel_tapas_mutual $SRC_DIR/parallel_tapas_mutual_nw


    # Build single-threaded version
    echoCyan env CC=${CC} CXX=${CXX} MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" make VERBOSE=1 MODE=debug -C $SRC_DIR tapas
    env CC=${CC} CXX=${CXX} MPICC="${MPICC}" MPICXX="${MPICXX}" make VERBOSE=1 MODE=debug -C $SRC_DIR tapas
}

function accuracyCheck() {
    MAX_ERR=5e-3

    local fname=$1
    PERR=$(grep "Rel. L2 Error" $fname | grep pot | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")
    AERR=$(grep "Rel. L2 Error" $fname | grep acc | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")

    echo "PERR='$PERR'"
    echo "AERR='$AERR'"

    if [[ $(python -c "print(float('$PERR') < $MAX_ERR)") != "True" ]]; then
        echoRed "*** Error check failed. L2 Error (pot) $PERR > $MAX_ERR"
        STATUS=$(expr $STATUS + 1)
    else
        echoGreen pot check OK
    fi
    if [[ $(python -c "print(float('$AERR') < $MAX_ERR)") != "True" ]]; then
        echoRed "*** Error check failed. L2 Error (acc) $AERR > $MAX_ERR"
        STATUS=$(expr $STATUS + 1)
    else
        echoGreen acc check OK
    fi
}

function tapasCheck() {
    echo --------------------------------------------------------------------
    echo ExaFMM
    echo --------------------------------------------------------------------

    build_fmm

    for ts in 1 2 3; do
    for nb in ${NB[@]}; do
    for ncrit in ${NCRIT[@]}; do
    for dist in ${DIST[@]}; do
    for mutual in "" "_mutual"; do
    for opt in "" "_mt" "_nw"; do
    for np in ${NP[@]}; do
        rm -f $TMPFILE; sleep 0.5s

        BIN=$SRC_DIR/parallel_tapas${mutual}${opt}

        if [[ -x ${BIN} ]]; then
            # run Exact LET TapasFMM
            rm -f $TMPFILE; sleep 0.5s # make sure that file is deleted on NFS
            echoCyan ${MPIEXEC} -n $np $BIN -n $nb -c $ncrit -d $dist -r ${ts}
            ${MPIEXEC} -n $np $BIN -n $nb -c $ncrit -d $dist -r ${ts} > $TMPFILE
            echo "exit status=$?"
            echo "TMPFILE=${TMPFILE}"
            cat $TMPFILE ||:

            accuracyCheck $TMPFILE
        else
            echo "*** Skipping ${BIN}"
        fi
        echo
        echo
    done
    done
    done
    done
    done
    done
    done
}

tapasCheck

# Check some special cases

echo
echo --------------------------------------------------------------------
echo "ExaFMM (with Ncrit > Nbodies)"
echo --------------------------------------------------------------------
echo

for MUTUAL in "" "_mutual" ; do
    rm -f $TMPFILE; sleep 0.5s
    echoCyan ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 1000 -c 1024 -d c
    ${MPIEXEC} -np 1 $SRC_DIR/parallel_tapas${MUTUAL} -n 1000 -c 1024 -d c  > $TMPFILE
    cat $TMPFILE ||:
    
    accuracyCheck $TMPFILE
done

if [[ $STATUS -eq 0 ]]; then
    echo OK.
else
    echoRed "***** Test failed."
fi

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
        for nb in ${NB[@]}; do
            for ncrit in ${NCRIT[@]}; do
                echoCyan ${MPIEXEC} -n 1 ./$BIN --numBodies 1000
                ${MPIEXEC} -n 1 ./$BIN --numBodies 10000 > $TMPFILE
                cat $TMPFILE
                accuracyCheck $TMPFILE
            done
        done
    done

fi
