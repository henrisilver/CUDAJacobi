#!/bin/bash

let i=0

dimensions=(10 100 500 1000 1500 5000 10000)

for n in "${dimensions[@]}"
do
	echo Generating input$i.in with $n dimension
    ./input_generator $n input$i.in
    let i=i+1
done
