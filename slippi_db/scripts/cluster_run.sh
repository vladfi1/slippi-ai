#!/usr/bin/env bash

CLUSTER_FILE=$1
SCRIPT=$2
shift 2

# make sure we tear down the cluster even if the script fails
set +e

ray submit --start $CLUSTER_FILE $SCRIPT "$@"
ray down -y $CLUSTER_FILE
