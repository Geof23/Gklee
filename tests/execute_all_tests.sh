#!/bin/bash
./clean.sh
for D in `find . -type d -not -name "."`
do
    echo $D
    cd $D
    ../execute_test.sh *.cu
    cd ..
    echo 
done