#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG_ONLY=$1
CONFIG_INTERN=$2
GPUS=$3

./$(dirname "$0")/dist_train.sh $CONFIG_ONLY $GPUS ${@:4}

if [ $? -eq 0 ]; then
    $PYTHON $(dirname "$0")/crop_backplane.py
    if [ $? -eq 0 ]; then
        ./$(dirname "$0")/dist_train.sh $CONFIG_INTERN $GPUS ${@:4}
        if [ $? -ne 0 ]; then
            echo "intern model train failed"
        fi
    else
        echo "crop failed"
    fi
else
   echo "backplane model train failed\n"
fi
