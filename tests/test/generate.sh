#!/bin/bash

let i=0

dimensions=(10 100 1000 10000)

for n in "${dimensions[@]}"
	echo Generating input$i.in with matrix dimension equal to $n
    ./input_generator $n input$i.in
    let i=i+1
done
