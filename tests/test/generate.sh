#!/bin/bash

let i=0

dimensions=(10 100 1000 10000)

for n in "${dimensions[@]}"
	echo Generating inpug$i.in with $n dimension
    ./input_generator $n input$i.in
    let i=i+1
done
