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

#define DEBUG 1
#define DEBUG_LEVEL_2 0

// Os números que compõem a matriz gerada aleatóriamente terão
// valores entre -1024 e 1024
#define MAXVAL 1024
#define ERROR_TOLERANCE 0.0001
#define null NULL

void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int *n, FILE *file);
float absolute(float x);
int doesNotConverge(float **A, int n);
void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile);
void normalize(float **A, float *currentX, float *B, float **normalizedA, float *normalizedB ,int n);
void copyCurrentXToPreviousX(float *currentX, float *previousX, int n);
void computeNewCurrentX(float *currentX, float *previousX, float **normalizedA, float *normalizedB, int n);
float getError(float *currentX, float *previousX, int n);
void showResults(float **A, float *currentX, float *B, int n);
void printAll(float **A, float *X, float *B, int n);
void cleanUp(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int n);

int main(int argc, const char * argv[]) {

    FILE *inputFile = null; 
    FILE *outputFile = null;
    float **A; // Matriz A original
    float *previousX; // Vetor X - variáveis - valores da iteração anterior
    float *currentX; // Vetor X - variáveis - valores da iteração atual
    float *B; // Vetor B original
    float **normalizedA; // Matriz A normalizada
    float *normalizedB; // Vetor B normalizado
    int n; // Ordem da matriz A
    clock_t start, end;
    double cpu_time_used;
    int i;

    inputFile = fopen(argv[1],"rb");
    if (inputFile == null) {
        perror("Failed to open file");
        exit(0);
    }

    outputFile = fopen(argv[2],"wb");
    if (outputFile == null) {
        perror("Failed to open file");
        exit(0);
    }
     
    start = clock();
    initialize(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, &n, inputFile);
    readDataFromInputFile(A, B, n, inputFile);
    normalize(A, currentX, B, normalizedA, normalizedB, n);
    //printAll(A,currentX,B,n);

    //printf("\n\n***** Jacobi-Richardson Method Sequential Execution *****\n");
    do {
        copyCurrentXToPreviousX(currentX, previousX, n);
        computeNewCurrentX(currentX, previousX, normalizedA, normalizedB, n);
    } while(getError(currentX, previousX, n) > ERROR_TOLERANCE);
    end = clock();

    //printf("\n\n");
    //showResults(A, currentX, B, n);
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %fs for dimension %d\n", cpu_time_used, n);

    for(i = 0; i < n; i++) {
        fwrite(&currentX[i], sizeof(float), 1, outputFile);
    }

    fclose(inputFile);
    fclose(outputFile);
    cleanUp(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, n);
    return 0;
}

// Função para alocar matrizes e vetores
void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB , int *n, FILE *file) {
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

float absolute(float x) {
    //
    return x < 0.0 ? -x : x;
}

void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile) {
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

// Calcula os valores normalizados para a matriz A e para o vetor B.
// Os valores iniciais das variáveis X são definidos como o vetor B normalizado.
void normalize(float **A, float *currentX, float *B, float **normalizedA, float *normalizedB ,int n) {
    int i, j;

    for(i = 0; i < n; i ++) {
        for(j = 0; j < n; j++) {
            if(i == j) {
                normalizedA[i][j] = 0.0;
            }
            else {
                normalizedA[i][j] = A[i][j] / A[i][i];
            }
        }
    }

    for(i = 0; i < n; i++) {
        normalizedB[i] = B[i] / A[i][i];
        currentX[i] = normalizedB[i];
    }
}

// Função utilizada para copiar os valores atuais do vetor X para o vetor
// de valores anteirores, já que estamos avançando para a próxima iteração.
void copyCurrentXToPreviousX(float *currentX, float *previousX, int n) {
    int i;
    for(i = 0; i < n; i++) {
        previousX[i] = currentX[i];
    }
}

// Função de cálculo dos novos valores de X.
void computeNewCurrentX(float *currentX, float *previousX, float **normalizedA, float *normalizedB, int n) {
    int i, j;
    float sum;
    for(i = 0; i < n; i++) {
        sum = 0.0;
        for(j = 0; j < n; j++) {
            if(i != j) {
                sum -= normalizedA[i][j] * previousX[j];
            }
        }
        sum += normalizedB[i];
        currentX[i] = sum;
    }
}

// Função utilizada para calcular o maior erro relativo
// obtido na iteração atual do método. São calculados os
// maiores valores absolutos da diferença de cada X[i]
// (atual menos o anterior) e os valores absolutos de cada X[i]
// atual. O retorno é a razão entre esses dois valores.
float getError(float *currentX, float *previousX, int n) {
    float maxRelativeError;
    float currentAbsoluteError;
    float maxAbsoluteEntry;
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

    if (DEBUG) {
        printf("getError - maxRelativeError [%f]\n", maxRelativeError);
    }
    return maxRelativeError;
}

// Função para exibir os resultados obtidos. É escolhida aleatoriamente
// uma equação e são mostrados o resultado estimado e o esperado em B.
void showResults(float **A, float *currentX, float *B, int n) {
    int i;
    float calculatedResult = 0.0;

    int line = rand() % n;
    /*
    printf("Resultado X:\n");
    
    for(i = 0; i < n; i++) {
        printf("[%2.3f] ", currentX[i]);
    }
    printf("\n\n");
    */
    printf("Equação aleatória para avaliação de corretude:\n");
    for (i = 0; i < n; i++) {
        //printf("%2.3f * %2.3f", A[line][i], currentX[i]);
        calculatedResult += A[line][i] * currentX[i];
        /*if(i != n-1) {
            printf(" + ");
        }
        else {
            printf(" = [%2.3f]\n", calculatedResult);
        }*/
    }
    printf("Valor esperado para o resultado:\n%2.3f\n", B[line]);
    printf("Diferença entre resultados:\n%2.3f\n", B[line] - calculatedResult);
}

// Imprime os valores das matrizes
void printAll(float **A, float *X, float *B, int n) {
    printf("\nA:\n");
    
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%f ", A[i][j]);
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

// Libera memória alocada dinamicamente
void cleanUp(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int n) {
    int i;
    for(i = 0; i < n; i++) {
        free((*A)[i]);
    }
    free(*A);
    
    for(i = 0; i < n; i++) {
        free((*normalizedA)[i]);
    }
    free(*normalizedA);
    
    free(*B);
    free(*normalizedB);
    free(*currentX);
    free(*previousX);
}
