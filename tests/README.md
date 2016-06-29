Test framework roadmap:

1) Compile the test generator file using

gcc test/testgenerator.c -o test/generator

Then go to /test directory and create the test cases by running

cd test/
./generate.sh

You may change the width, height and frames values in the .sh file as you please

--------------------------------------------------------------------------------

2) Go to the /stats directory and compile the metric analyzer

cd stats/
gcc metrics.c -o metrics-analyzer

--------------------------------------------------------------------------------

3) Go back to the root directory and compile the sequential solution, placing the executable file in the parallel folder.
Name the executable "smooth"

cd ..
gcc common/smooth.c sequential/solution.c -o sequential/smooth -Isequential -fopenmp

--------------------------------------------------------------------------------

4) Compile the parallel solution, placing the executable file in the parallel folder.
Name the executable "smoothparallel"

gcc common/smooth.c parallel/solution.c -o parallel/smoothparallel -Iparallel -fopenmp

--------------------------------------------------------------------------------

5) We're all set and ready to run (and compare) the solutions!
To run the test cases and get outputs, run:

./run.sh

Your outputs will be in the respective sequential/parallel directories.
The execution time tables will be inside the /stats directory.

6) Run the metrics analyzer on the execution time tables:

cd stats/
./getmetrics.sh

The output of the metrics analyzer will be in a file named overallmetrics.mtr inside the /stats folder.

7) Optionally, clean the sequential and parallel directories (remove outputs) using:

cd ..
./cleanup.sh

And clean the test directory

cd ../test/
./remove.sh

---------------------------------------------------------------------------------

If any .sh script does not run, try the command:

chmod +x file.sh

---------------------------------------------------------------------------------

Configuration in a nutshell:

Run from the root directory:
./config.sh

or copy and paste the following code a line at a time:

gcc test/testgenerator.c -o test/generator
cd stats/
gcc metrics.c -o metrics-analyzer
cd ..
gcc common/smooth.c sequential/solution.c -o sequential/smooth -Isequential -fopenmp
gcc common/smooth.c parallel/solution.c -o parallel/smoothparallel -Iparallel -fopenmp

Execution in a nutshell:

Run from the root directory:
./completerun.sh

cd test/
./generate.sh
cd ../
./run.sh
cd stats/
./getmetrics.sh
cd ..
./cleanup.sh
