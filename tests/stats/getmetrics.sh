#!/bin/bash

echo Generating metric results...

./metrics-analyzer sequentialMetrics.mtr parallelMetrics.mtr comparison.mtr

echo You can check the metric results in comparison.mtr
