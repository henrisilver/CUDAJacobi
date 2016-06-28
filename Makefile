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
	gcc -o paralell include/src/jacobi-richardson.c

runseq:
	./generator 1000 input
	./sequential input
