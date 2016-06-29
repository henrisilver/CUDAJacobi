#!/bin/bash

cd test/
./generate.sh
cd ../
./run.sh
#cd stats/
#./getmetrics.sh
cd ../
./cleanup.sh
