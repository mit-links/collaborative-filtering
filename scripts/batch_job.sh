#!/bin/sh
# the batch job executed
FOLDER_NAME_REMOTE=$1 # the first parameter should be the remote folder name

START_PYTHON_PATH=$FOLDER_NAME_REMOTE/scripts
# this script is submitted as the batch job

cd $START_PYTHON_PATH
PYTHONPATH=../.. python3 Start.py ../config/config.json

