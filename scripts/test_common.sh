#!/bin/sh
# test_common.sh
# Sourced by test shell scripts in sample/ subdirectoires.
# Provides a few common variables
#
# defined variables:
#   TMPFILE
#   SCALE
#   MPICXX

unset TMPFILE
set -u
set -e

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

atexit() {
    [[ -n "${TMPFILE-}" ]] && rm -f "$TMPFILE"
}

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

function test_start() {
    local DATE=$(date)
    local HOST=$(hostname)
    echo "Test script '$0' started."
    echo "Date: $DATE"
    echo "Host: $HOST"
}

# -------------------------------------------
#   Setup MPICXX var
# -------------------------------------------

if [[ -z "${CXX:-}" ]]; then
    CXX=g++
fi

if [[ -z "${MAKE:-}" ]]; then
    MAKE=make
fi

echo "-----------------------------------------------"
echo "MPICXX=${MPICXX}"
echo "MAKE=${MAKE}"
echo "-----------------------------------------------"

echo MAKE=${MAKE}
