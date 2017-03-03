#!/bin/bash
# Run unit tests of Tapas framework

#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

export TAPAS_REPORT_FILENAME=""

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
TAPAS_DIR=$(cd ${TEST_DIR}/..; pwd)
#TAPAS_DIR=$(pushd $FMM_DIR/../../ >/dev/null ; pwd -P; popd >/dev/null)

echo TEST_DIR="$TEST_DIR"
echo TAPAS_DIR="$TAPAS_DIR"

cd ${TEST_DIR}

source $TAPAS_DIR/scripts/test_common.sh

echo --------------------------------------------------------------------
echo C++ Unit Tests
echo --------------------------------------------------------------------

echoCyan make MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $TEST_DIR clean
make -j MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $TEST_DIR clean

echoCyan make MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $TEST_DIR all
make -j MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $TEST_DIR all &&:
echo $?

TEST_TARGETS=$(make -C $TEST_DIR list | grep -v make | grep -v echo | grep test_)
for t in $TEST_TARGETS; do
    for NP in 1 2 3 4; do
        echoCyan ${MPIEXEC} -np $NP $TEST_DIR/$t
        ${MPIEXEC} -np $NP $TEST_DIR/$t
    done
done


