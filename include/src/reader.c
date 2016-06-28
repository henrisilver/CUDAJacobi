#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXVAL 1024

int main(int argc, char const *argv[]) {
    FILE *file = NULL;
    int dimension = 0;
    int i;
    int j;
    float **A = NULL;
    float *B = NULL;

    file = fopen("out","rb");
    if(file == NULL){
        perror("Faio");
        exit(0);
    }


    fread(&dimension,sizeof(int),1,file);

    A = (float **) calloc(sizeof(float *), dimension);
    B = (float *) calloc(sizeof(float), dimension);
    for (i = 0 ; i < dimension ; i++){
        A[i] = (float *) calloc(sizeof(float), dimension);
    }

    printf("Dimension: %d\n", dimension);

    printf("Matriz A:\n");
    for (i = 0 ; i < dimension ; i++){
        for (j = 0 ; j < dimension ; j++){
            fread(&A[i][j], sizeof(float), 1, file);
            printf("[%f] ", A[i][j]);
        }
        printf("\n");
    }

    printf("Betor V:\n");
    for (i = 0 ; i < dimension ; i++){
        fread(&B[i], sizeof(float), 1, file);
        printf("[%f] ", B[i]);
    }
    printf("\n");
    fclose(file);

    return 0;
}