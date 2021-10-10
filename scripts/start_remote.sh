#!/bin/sh
NETHZ=myNethz
FOLDER_NAME_LOCAL=collab_filtering
ID=$(shuf -i 0-100000000 -n 1)
FOLDER_NAME_REMOTE=$FOLDER_NAME_LOCAL-$ID
REMOTE_SCRIPT_NAME=exec_on_euler.sh
BATCH_SCRIPT_NAME=batch_job.sh

chmod +x $REMOTE_SCRIPT_NAME
chmod +x $BATCH_SCRIPT_NAME
echo "job has id $ID"

scp -r ../../$FOLDER_NAME_LOCAL $NETHZ@euler.ethz.ch:$FOLDER_NAME_REMOTE # copy entire folder
ssh $NETHZ@euler.ethz.ch "$FOLDER_NAME_REMOTE/scripts/$REMOTE_SCRIPT_NAME $FOLDER_NAME_REMOTE $ID" # execute remote script
