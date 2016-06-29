#!/bin/bash

echo Executing sequential tests
#echo ""

let i=0
for filename in test/*.in;
do
    echo -e "\t$filename"
    sequential/sequential $filename sequential/output$i.out >> stats/sequentialMetrics.mtr
    let i=i+1
done

echo ""
echo Executing parallel tests
#echo ""

let i=0
for filename in test/*.in;
do
    echo -e "\t$filename"
    parallel/parallel $filename parallel/output$i.out >> stats/parallelMetrics.mtr
    let i=i+1
done

#mv test/*.out parallel/

echo ""
echo See output files in the sequential and parallel folders for results

echo ""
