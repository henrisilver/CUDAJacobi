/*
 * SSC0742 - Programação Concorrente
 * Professor Paulo Sérgio Lopes de Souza
 * Trabalho Prático 4 - Solução Sequencial para um sistema linear utilizando o método de Jacobi-Richardson
 * Grupo 03
 * Integrantes:
 * -> Adriano Belfort de Sousa ­- 7960706
 * -> Giuliano Barbosa Prado -­ 7961109
 ­* -> Henrique de Almeida Machado da Silveira -­ 7961089
 ­* -> Marcello de Paula Ferreira Costa ­- 7960690
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DOES_NOT_CONVERGE 1
#define CONVERGE 0

#define DEBUG 0
#define DEBUG_LEVEL_2 0

// Os números que compõem a matriz gerada aleatóriamente terão
// valores entre -1024 e 1024
#define MAXVAL 1024
#define ERROR_TOLERANCE 0.0001
#define null NULL

 __device__ int reachedErrorTolerance = 0;

 __device__ float absolute(float x) {
    
    return x < 0.0 ? -x : x;
}

 __device__ void normalize(float *A, float *currentX, float *B, float *normalizedA, float *normalizedB ,int n) {
    int i, j;

    for(i = 0; i < n; i ++) {
        for(j = 0; j < n; j++) {
            if(i == j) {
                normalizedA[i * n + j] = 0.0;
            }
            else {
                normalizedA[i * n + j] = A[i * n + j] / A[i * n + i];
            }
        }
    }

    for(i = 0; i < n; i++) {
        normalizedB[i] = B[i] / A[i * n + i];
        currentX[i] = normalizedB[i];
    }
}

__device__ void getError(float *currentX, float *previousX, int n) {
    float maxRelativeError;
    float currentAbsoluteError;
    float currentRelativeError;
    float currentEntry;
    int i;

    currentAbsoluteError = absolute(currentX[0] - previousX[0]);
    currentEntry = absolute(currentX[0]);
    currentRelativeError = currentAbsoluteError/currentEntry;
    maxRelativeError = currentAbsoluteError;

    for(i = 1; i < n; i++) {
        currentAbsoluteError = absolute(currentX[i] - previousX[i]);
        currentEntry = absolute(currentX[i]);
        currentRelativeError = currentAbsoluteError/currentEntry;
        if (currentRelativeError > maxRelativeError){
            maxRelativeError = currentRelativeError;
        }
    }

    // if (DEBUG) {
    //     printf("getError - maxRelativeError [%f]\n", maxRelativeError);
    // }

    if(maxRelativeError < ERROR_TOLERANCE) {
        reachedErrorTolerance = 1;
    }
}

 __device__ void computeNewCurrentX(float *currentX, float *previousX, float *normalizedA, float *normalizedB, int n) {
    
    int myIndex = threadIdx.x;
    int j;
    float sum;

    sum = 0.0;
    for(j = 0; j < n; j++) {
        if(myIndex != j) {
            sum -= normalizedA[myIndex * n + j] * previousX[j];
        }
    }
        sum += normalizedB[myIndex];
        currentX[myIndex] = sum;
        __syncthreads();

}

 __device__ void copyCurrentXToPreviousX(float *currentX, float *previousX) {
    
    int myIndex = threadIdx.x;
    previousX[myIndex] = currentX[myIndex];
}

 __global__ void solveJacobiRichardson(float *A, float *B, float *normalizedA, float *normalizedB, float * currentX, float *previousX, int n) {

    int myIndex = threadIdx.x;
    if(myIndex < n) {

        if(myIndex == 0) {
            normalize(A, currentX, B, normalizedA, normalizedB, n);
        }

        do {
            copyCurrentXToPreviousX(currentX, previousX);
            computeNewCurrentX(currentX, previousX, normalizedA, normalizedB, n);
            if(myIndex == 0) {
                getError(currentX, previousX, n);
            }
        } while(reachedErrorTolerance == 0);
    }

 }

__host__ void initialize(float **A, float **currentX, float **B, int *n, FILE *file) {
    fread(n, sizeof(int), 1, file);

    *A = (float *) malloc((*n) * (*n) * sizeof(float));
    *currentX = (float *) malloc(*n * sizeof(float));
    *B = (float *) malloc(*n * sizeof(float));

}

__host__ void readDataFromInputFile(float *A, float *B, int n, FILE *inputFile) {
    int i, j;

    for(i = 0; i < n; i ++) {
        for(j = 0; j < n; j++) {
            fread(&A[i * n + j], sizeof(float), 1, inputFile);
        }
    }

    for(i = 0; i < n; i ++) {
        fread(&B[i], sizeof(float), 1, inputFile);
    }
}

__host__ void showResults(float *A, float *currentX, float *B, int n, FILE *outputFile) {
    int i;
    float calculatedResult = 0.0;
    int line = rand() % n;
        
    for(i = 0; i < n; i++) {
        fprintf(outputFile, "X[%d] = %f\n", i, currentX[i]);
    }
    
    fprintf(outputFile, "\nEquação aleatória para avaliação de corretude:\n");
    for (i = 0; i < n; i++) {
        fprintf(outputFile, "%2.3f * %2.3f", A[line * n + i], currentX[i]);
        calculatedResult += A[line * n + i] * currentX[i];
        if(i != n-1) {
            fprintf(outputFile, " + ");
        }
        else {
            fprintf(outputFile, " = [%2.3f]\n", calculatedResult);
        }
    }
    fprintf(outputFile, "Valor esperado para o resultado:\n%2.3f\n", B[line]);
    fprintf(outputFile, "Diferença entre resultados:\n%2.3f\n", B[line] - calculatedResult);
}

__host__ void printAll(float *A, float *X, float *B, int n) {
    printf("\nA:\n");
    
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
    
    printf("\nX:\n");
    
    for(i = 0; i < n; i++) {
        printf("%f ", X[i]);
    }
    printf("\n");
    
    printf("\nB:\n");
    for(i = 0; i < n; i++) {
        printf("%f ", B[i]);
    }
    printf("\n");
}

__host__ void cleanUp(float *h_A, float *h_currentX, float *h_B, float *d_A, float *d_currentX, float *d_B, float *d_normalizedA, float *d_previousX, float *d_normalizedB) {
    free(h_A);
    free(h_B);
    free(h_currentX);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_currentX);
    cudaFree(d_normalizedA);
    cudaFree(d_normalizedB);
    cudaFree(d_previousX);
    
}

int main(int argc, const char * argv[]) {

    FILE *inputFile = null;   
    FILE *outputFile = null;
    float *h_A; // Matriz A original
    float *h_currentX; // Vetor X - variáveis - valores da iteração atual
    float *h_B; // Vetor B original
 
    int n; // Ordem da matriz A

    float *d_A;
    float *d_currentX;
    float *d_B;
    float *d_previousX; // Vetor X - variáveis - valores da iteração anterior
    float *d_normalizedA; // Matriz A normalizada
    float *d_normalizedB; // Vetor B normalizado

    clock_t start, end;
    double cpu_time_used;

    inputFile = fopen(argv[1],"rb");
    if (inputFile == null) {
        perror("Failed to open file");
        exit(0);
    }

    outputFile = fopen(argv[2],"wt");
    if (outputFile == null) {
        perror("Failed to open file");
        exit(0);
    }


    start = clock();

    initialize(&h_A, &h_currentX, &h_B, &n, inputFile);
    readDataFromInputFile(h_A, h_B, n, inputFile);

    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_currentX, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_previousX, n * sizeof(float));
    cudaMalloc(&d_normalizedA, n * n * sizeof(float));
    cudaMalloc(&d_normalizedB, n * sizeof(float));

    cudaMemcpy(d_A,h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    solveJacobiRichardson<<<1, n>>>(d_A, d_B, d_normalizedA, d_normalizedB, d_currentX, d_previousX, n);

    cudaMemcpy(h_currentX,d_currentX, n * sizeof(float),cudaMemcpyDeviceToHost);

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %fs for dimension %d\n", cpu_time_used, n);

    fprintf(outputFile, "*** Parallel Results ***\n");
    showResults(h_A, h_currentX, h_B, n, outputFile);

    fclose(inputFile);
    fclose(outputFile);
    
    cleanUp(h_A, h_currentX, h_B, d_A, d_currentX, d_B, d_normalizedA, d_previousX, d_normalizedB);

    return 0;
}
