all:
	gcc -o generator include/src/input_generator.c
	gcc -o reader include/src/reader.c
	gcc -o sequential include/src/sequential.c

generator:
	gcc -o generator include/src/input_generator.c

reader:
	gcc -o reader include/src/reader.c

clean:
	rm reader generator in

sequential:
	gcc -o sequential include/src/sequential.c

paralell:
	nvcc -o paralell include/src/parallel.cu

runseq:
	./generator 20000 input
	./sequential input
