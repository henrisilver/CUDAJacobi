#!/bin/bash

echo Executing sequential tests
#echo ""

let i=0
for filename in test/*.in;
do
    echo -e "\t$filename"
    sequential/sequential test/$filename sequential/output$i.out >> stats/sequentialMetrics.mtr
    let i=i+1
done

echo ""
echo Executing parallel tests
#echo ""

let i=0
for filename in test/*.in;
do
    echo -e "\t$filename"
    parallel/parallel test/$filename parallel/output$i.out >> stats/parallelMetrics.mtr
    let i=i+1
done

#mv test/*.out parallel/

echo ""
echo Scanning for output differences...

for sequentialResult in $(find sequential/ -name '*.out');
do
    sfname=$(basename $sequentialResult)
    #echo $sfname

    for parallelResult in $(find parallel/ -name '*.out');
    do
        pfname=$(basename $parallelResult)
        #echo $pfname

        if [ "$sfname" = "$pfname" ];
        then
            diff $sequentialResult $parallelResult
        fi
    done
done

echo ""
