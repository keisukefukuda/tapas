#!/bin/sh
# Run tests for TapasFMM application.

#----------------------------------------------------------
# functions


#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

FMM_DIR=$(dirname $0)
TAPAS_DIR=$(pushd $FMM_DIR/../../ >/dev/null ; pwd -P; popd >/dev/null)

echo FMM_DIR="$FMM_DIR"
echo TAPAS_DIR="$TAPAS_DIR"

cd ${FMM_DIR}/examples

source $TAPAS_DIR/scripts/test_common.sh

#----------------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------------

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

function build() {
    MAKE_FLAGS="MODE=release"

    if [[ -d "${MYTH_DIR:-}" ]]; then
        MAKE_FLAGS="${MAKE_FLAGS} MTHREAD=1 MYTH_DIR=\"${MYTH_DIR}\""
    fi

    if [[ "${TWOSIDE_LET:-}" == "1" ]]; then
        MAKE_FLAGS="${MAKE_FLAGS} USE_TWOSIDE_LET=1"
    fi

    if [[ "${USE_WEIGHT:-}" == "0" ]]; then
        MAKE_FLAGS="${MAKE_FLAGS} WEIGHT=0"
    fi

    echoCyan env MPICXX=${MPICXX} CXX=${CXX} ${MAKE} ${MAKE_FLAGS} VERBOSE=1 clean tapas
    env MPICXX=${MPICXX} CXX=${CXX} ${MAKE} ${MAKE_FLAGS} VERBOSE=1 clean tapas &&:
}

function run() {
    rm -f "${TMPFILE}"
    
    # Run the non-mutual version
    echoCyan ${MPIEXEC} -n ${NP} ./parallel_tapas \
             --numBodies ${NB} --ncrit=${NC} --dist ${D}
    ${MPIEXEC} -n ${NP} ./parallel_tapas \
               --numBodies ${NB} --ncrit=${NC} --dist ${D} \
               >${TMPFILE} &&:
    local STAT="$?"
    echo exit status $STAT
    
    if [[ "${STAT}" != 0 ]]; then
        echoRed "ERROR: program filed with exit code ${STAT}"
        cat $TMPFILE ||:
    else
        if [[ "${VERBOSE:-}" > 0 ]]; then
            cat -u $TMPFILE &&:
            echo "cat status=$?"
        fi
    fi
    accuracyCheck $TMPFILE

    # Run the mutual version
    echoCyan ${MPIEXEC} -n ${NP} ./parallel_tapas_mutual \
             --numBodies ${NB} --ncrit=${NC} --dist ${D}
    ${MPIEXEC} -n ${NP} ./parallel_tapas_mutual \
               --numBodies ${NB} --ncrit=${NC} --dist ${D} \
               >${TMPFILE} &&:
    local STAT="$?"
    echo exit status $STAT

    if [[ "${STAT}" != 0 ]]; then
        echoRed "ERROR: program filed with exit code ${STAT}"
        cat $TMPFILE ||:
    else
        if [[ "${VERBOSE:-}" > 0 ]]; then
            cat $TMPFILE &&:
            echo "cat status=$?"
        fi
    fi
    accuracyCheck $TMPFILE
}

#----------------------------------------------------------------------------
# Main part
#----------------------------------------------------------------------------

test_start

STATUS=0

for EQUATION in Laplace; do
for BASIS in Spherical; do
for WEIGHT in 0 1; do
for TWOSIDE_LET in 0 1; do

build
    
for NP in 1 2 4; do
for NB in 1000 2000 4000; do
for NC in 32 64; do
for D in c s l; do # Skip plummer distribution for now because the trg-2-side and trg-1-side LET inspector takes very long.
    echo "EQUATION=$EQUATION BASIS=${BASIS} WEIGHT=${WEIGHT} TWOSIDE_LET=${TWOSIDE_LET} NB=${NB} NP=${NP} NC=${NC} D=${D}"
    run
done
done
done
done

done
done
done
done

echo status=$STATUS

if [[ "${STATUS}" != 0 ]]; then
    echoRed "${STATUS} tests failed."
fi

exit $STATUS
