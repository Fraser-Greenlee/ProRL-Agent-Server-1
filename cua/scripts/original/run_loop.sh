#!/bin/bash

for i in {1..20}
do
   echo "--- Starting Iteration $i ---"
   NUM_ACTORS=4 bash run.sh
   sleep 600
done
