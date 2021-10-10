#!/bin/sh
# this script is executed remotely on euler

FOLDER_NAME_REMOTE=$1 # the first parameter should be the remote folder name
ID=$2

BATCH_JOB_SCRIPT_PATH=$FOLDER_NAME_REMOTE/scripts/batch_job.sh
OUT_FOLDER=~/out

if [ ! -d $OUT_FOLDER ]; then # if folder doesn't exist
	mkdir $OUT_FOLDER # folder may already exist but not a problem
fi

#load dependencies
module load new gcc/4.8.2 python/3.6.0

bsub -n 4 -W 3:59 -oo $OUT_FOLDER/$ID.out "$BATCH_JOB_SCRIPT_PATH $FOLDER_NAME_REMOTE"


