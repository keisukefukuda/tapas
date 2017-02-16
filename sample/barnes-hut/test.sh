#!/bin/sh
set -eu

BH_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
TAPAS_DIR=$(cd ${BH_DIR}/../../; pwd)
#TAPAS_DIR=$(pushd $FMM_DIR/../../ >/dev/null ; pwd -P; popd >/dev/null)

echo BH_DIR=${BH_DIR}
echo TAPAS_DIR="${TAPAS_DIR}"

cd $BH_DIR

echo source $TAPAS_DIR/scripts/test_common.sh
source $TAPAS_DIR/scripts/test_common.sh

BIN=$BH_DIR/bh

NP=(1 4)
NB=(1000 2000 4000)
MAX_ERR=1e-2

echoCyan make CXX=\"${CXX}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=debug -C $BH_DIR clean $(basename $BIN)
make CXX=${CXX} MPICXX="${MPICXX}" VERBOSE=1 MODE=debug -C $BH_DIR clean $(basename $BIN)

for np in ${NP[@]}; do
    for nb in ${NB[@]}; do
        echoCyan ${MPIEXEC} -n $np $BIN -w $nb
        ${MPIEXEC} -n $np $BIN -w $nb >$TMPFILE 2>&1

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
