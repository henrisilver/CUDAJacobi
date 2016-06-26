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

// Gerando números aleatórios para esse trabalho entre -1024 e 1024.
#define MAXVAL 1024
#define ERROR_TOLERANCE 0.001

// Função para alocar matrizes e vetores
void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int n) {
    int i; // Variável utilizada para iteração
    
    *A = (float **) malloc(n * sizeof(float *));
    for(i = 0; i < n; i++) {
        (*A)[i] = (float *) malloc(n * sizeof(float));
    }
    
    *normalizedA = (float **) malloc(n * sizeof(float *));
    for(i = 0; i < n; i++) {
        (*normalizedA)[i] = (float *) malloc(n * sizeof(float));
    }
    
    *currentX = (float *) malloc(n * sizeof(float));
    
    *previousX = (float *) malloc(n * sizeof(float));
    
    *B = (float *) malloc(n * sizeof(float));
    *normalizedB = (float *) malloc(n * sizeof(float));
}

float absolute(float x) {
    return x < 0.0 ? -x : x;
}

// Checa se o método irá convergir ao analisar os valores de A.
// Análise feita por linhas.
// Se convergir, retorna 0. Se não convergir, retorna 1.
int doesNotConverge(float **A, int n) {
    int i, j;
    float sum;
    for(i = 0; i < n; i ++) {
        sum = 0.0;
        for(j = 0; j < n; j++) {
            if(i != j) {
                sum += absolute(A[i][j]);
            }
        }
        if(sum < absolute(A[i][i])) {
            return 1;
        }
    }
    
    return 0;
}

// Popula a matriz A e o vetor B com valores aleatórios
void populate(float **A, float *B, int n) {
    int i, j;
    srand((unsigned int) time(NULL));
    
    // Calcula a matriz A enquanto os valores aleatórios utilizados
    // forem tais que o método não converge
    do {
        for(i = 0; i < n; i ++) {
            for(j = 0; j < n; j++) {
                A[i][j] = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
            }
        }
    } while(doesNotConverge(A, n));
    
    // Popula o vetor B
    for(i = 0; i < n; i ++) {
        B[i] = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
    }
    
}

// Calcula os valores normalizados para a matriz A e para o vetor B.
// Os valores iniciais das variáveis X são definidos como o vetor B normalizado.
void calculateNormalizedValues(float **A, float *currentX, float *B, float **normalizedA, float *normalizedB ,int n) {
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
                sum += normalizedA[i][j] * previousX[j];
            }
        }
        sum += normalizedB[i];
        currentX[i] = sum;
    }
}

// Função utilizada para calcular o maior erro obtido na iteração atual do
// método. São calculados os maiores valores absolutos da diferença de
// cada X[i] (atual menos o anterior) e os valores absolutos de cada X[i]
// atual. O retorno é a razão entre esses dois valores.
float getError(float *currentX, float *previousX, int n) {
    float maxError, currentError, maxAbsolute;
    int i;
    maxError = absolute(currentX[0] - previousX[0]);
    maxAbsolute = absolute(currentX[0]);
    
    for(i = 1; i < n; i++) {
        currentError = absolute(currentX[i] - previousX[i]);
        if(maxError < currentError) {
            maxError = currentError;
        }
        if(maxAbsolute < absolute(currentX[i])) {
            maxAbsolute = absolute(currentX[i]);
        }
    }
    return maxError/maxAbsolute;
}

// Função para exibir os resultados obtidos. É escolhida aleatoriamente uma equação e são mostrados
// o resultado estimado e o esperado em B.
void showResults(float **A, float *currentX, float *B, int n) {
    int i;
    float calculatedResult = 0.0;
    int line = rand() % n;
    
    srand((unsigned int) time(NULL));
    
    printf("Resultados obtidos para X:\n");
    
    for(i = 0; i < n; i++) {
        printf("%f ", currentX[i]);
    }
    printf("\n");
    
    printf("Equação obtida: \n\n");
    for (i = 0; i < n; i++) {
        printf("%f * %f", A[line][i], currentX[i]);
        calculatedResult += A[line][i] * currentX[i];
        if(i != n-1) {
            printf(" + ");
        }
        else {
            printf(" = %f", calculatedResult);
        }
    }
    printf("\n\nValor esperado para o resultado: %f", B[line]);
    printf("\n\nDiferença entre resultados: %f\n\n", B[line] - calculatedResult);
    
    
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

int main(int argc, const char * argv[]) {
    
    float **A; // Matriz A original
    float *previousX; // Vetor X - variáveis - valores da iteração anterior
    float *currentX; // Vetor X - variáveis - valores da iteração atual
    float *B; // Vetor B original
    float **normalizedA; // Matriz A normalizada
    float *normalizedB; // Vetor B normalizado
    int n; // Ordem da matriz A
    
    scanf("%d", &n);
    
    initialize(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, n);
    populate(A, B, n);
    calculateNormalizedValues(A, currentX, B, normalizedA, normalizedB, n);
    
    do {
        copyCurrentXToPreviousX(currentX, previousX, n);
        computeNewCurrentX(currentX, previousX, normalizedA, normalizedB, n);
    } while(getError(currentX, previousX, n) > ERROR_TOLERANCE);
    
    showResults(A, currentX, B, n);
    
    cleanUp(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, n);
    
    return 0;
}
