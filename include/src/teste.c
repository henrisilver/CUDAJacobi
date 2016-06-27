#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXVAL 1024

int main(int argc, char const *argv[])
{
    FILE *file = NULL;
    int dimension = 0;
    int i;
    int j;
    float f;
    float **A = NULL;

    file = fopen("out","rb");
    if(file == NULL){
        perror("Faio");
        exit(0);
    }

    A = (float **) calloc(sizeof(float *), dimension);
    for (i = 0 ; i < dimension ; i++){
        A[i] = (float *) calloc(sizeof(float), dimension);
    }

    fread(&dimension,sizeof(int),1,file);
    printf("Dimension: %d\n", dimension);

    for (i = 0 ; i < dimension ; i++){
        for (j = 0 ; j < dimension ; j++){
            fread(&A[i][j], sizeof(float), 1, file);
        }
    }

    fclose(file);

    return 0;
}