#!/bin/bash
echo "compiling with nvcc"
nvcc -I $KLEE_HOME_DIR/Gklee/include $1 &> nvcc_log.txt

echo "running a.out"
./a.out &> a_out_log.txt

echo "compiling with gklee-nvcc"
options=""
if [  -e options.txt ]
then
    options=`head -1 options.txt | tr -d '\n'`
fi
gklee-nvcc -I $KLEE_HOME_DIR/Gklee/include $options  $1 &> gklee_nvcc_log.txt

echo "running with gklee" 
(time gklee -max-time=3600 -max-memory=2048 ${1:0:-3}) &> gklee_log.txt

echo "running with gkleep"
(time gklee -symbolic-config -max-time=3600 -max-memory=2048 ${1:0: -3}) &> gkleep_log.txt

echo "verifying output"
../verify.py &> results.txt

echo "updating master results"
cd ..
tail */results.txt > master_results.txt