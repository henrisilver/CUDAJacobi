#include <stdio.h>
#include <stdlib.h>

#define null NULL
/*
__global__ indica que é uma funcao do kernel que pode ser invocada do escopo de uma funcao
__host__ para ser executada na GPU
*/

/*
__device__ indica que e uma funcao do kernel e que somente pode ser chamada por outra
funcao kernel ou por outra __device__
*/
/*
__host__ funcao tradicional em C que so pode ser chamada por outra funcao __host__
essa funcao eh executada no host somente
*/

void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int *n, FILE *file);
void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile);
void printA(float **A, int n);
void printB(float *B, int n);


int main(int argc, char const *argv[]) {
    FILE *inputFile = null;   
    float **A; // Matriz A original
    float *previousX; // Vetor X - variáveis - valores da iteração anterior
    float *currentX; // Vetor X - variáveis - valores da iteração atual
    float *B; // Vetor B original
    float **normalizedA; // Matriz A normalizada
    float *normalizedB; // Vetor B normalizado
    int n; // Ordem da matriz A

    if (argc != 2) {
        fprintf(stderr, "Two arguments required! %d provided.\nusage: ./paralell <input_filename>\n", argc);
        exit(0);
    }

    inputFile = fopen(argv[1],"rb");
    if (inputFile == null) {
        perror("Failed to open file");
        exit(0);
    }

    initialize(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, &n, inputFile);
    readDataFromInputFile(A, B, n, inputFile);

    printA(A,n);
    printB(B,n);
    
    return 0;
}

void printA(float **A, int n) {
    int i, j;

    printf("\nA:\n");
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void printB(float *B, int n){
    int i;

    printf("\nB:\n");
    for(i = 0; i < n; i++) {
        printf("%f ", B[i]);
    }
    printf("\n");
}

// Função para alocar matrizes e vetores
__host __ void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB , int *n, FILE *file) {
    int i; // Variável utilizada para iteração
    fread(n, sizeof(int), 1, file);

    *A = (float **) malloc(*n * sizeof(float *));
    for(i = 0; i < *n; i++) {
        (*A)[i] = (float *) malloc(*n * sizeof(float));
    }

    *normalizedA = (float **) malloc(*n * sizeof(float *));
    for(i = 0; i < *n; i++) {
        (*normalizedA)[i] = (float *) malloc(*n * sizeof(float));
    }

    *currentX = (float *) malloc(*n * sizeof(float));
    *previousX = (float *) malloc(*n * sizeof(float));
    *B = (float *) malloc(*n * sizeof(float));
    *normalizedB = (float *) malloc(*n * sizeof(float));
}

__host __ void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile) {
    int i, j;

    for(i = 0; i < n; i ++) {
        for(j = 0; j < n; j++) {
            fread(&A[i][j], sizeof(float), 1, inputFile);
        }
    }

    for(i = 0; i < n; i ++) {
        fread(&B[i], sizeof(float), 1, inputFile);
    }
}