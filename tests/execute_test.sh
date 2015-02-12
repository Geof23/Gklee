#!/bin/bash

THIS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ -e `which nvcc` ] 
then
		echo "compiling with nvcc"
		nvcc -I $THIS_HOME/../Gklee/include $1 &> nvcc_log.txt

		echo "running a.out"
		./a.out &> a_out_log.txt
fi

echo "compiling with gklee-nvcc"
options=""
if [  -e options.txt ]
then
    options=`head -1 options.txt | tr -d '\n'`
fi
$THIS_HOME/../bin/gklee-nvcc -I $THIS_HOME/../Gklee/include $options  $1 &> gklee_nvcc_log.txt

echo "running with gklee" 
(time $THIS_HOME/../bin/gklee -max-time=3600 -max-memory=2048 ${1:0:-3}) &> gklee_log.txt

echo "running with gkleep"
(time $THIS_HOME/../bin/gklee -symbolic-config -max-time=3600 -max-memory=2048 ${1:0: -3}) &> gkleep_log.txt

echo "verifying output"
../verify.py &> results.txt

echo "updating master results"
cd ..
tail */results.txt > master_results.txt
