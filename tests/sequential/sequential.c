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

void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB ,int *n, FILE *file);
float absolute(float x);
int doesNotConverge(float **A, int n);
void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile);
void normalize(float **A, float *currentX, float *B, float **normalizedA, float *normalizedB ,int n);
void copyCurrentXToPreviousX(float *currentX, float *previousX, int n);
void computeNewCurrentX(float *currentX, float *previousX, float **normalizedA, float *normalizedB, int n);
float getError(float *currentX, float *previousX, int n);
void showResults(float **A, float *currentX, float *B, int n, FILE *outputFile);
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
    clock_t start; // Clock no comeco da execucao
    clock_t end; // Clock ao fim da execucao
    double cpu_time_used; // Tempo de cpu utilizado

    // Tenta abrir o arquivo de input e se falhar, sai do programa
    inputFile = fopen(argv[1],"rb");
    if (inputFile == null) {
        perror("Failed to open file");
        exit(0);
    }

    // Tenta abrir o arquivo de output e se falhar, sai do programa
    outputFile = fopen(argv[2],"wt");
    if (outputFile == null) {
        perror("Failed to open file");
        exit(0);
    }
    // Inicio da execucao
    start = clock();
    // Aloca a memoria para a matrix A, vetor B, matriz normalizada de A, matriz normalizada
    // de B, vetor X corrente e vetor X anterior. Além disso recupera do arquivo de entrada
    // a dimencao da matriz e dos vetores que serao lidos de inputFile
    initialize(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, &n, inputFile);

    // Apos a alocacao das memorias, readDataFromInputFile preenche a Matriz A e o vetor B
    // com os valores lidos de inputFile
    readDataFromInputFile(A, B, n, inputFile);

    // Calcula a matriz normalizada e o vetor normalizado
    // A matriz a vira (L* + R*)
    normalize(A, currentX, B, normalizedA, normalizedB, n);

    do {
        // Copia os valores atual de X para o proximo vetor X para a proxima iteracao
        // do metodo
        copyCurrentXToPreviousX(currentX, previousX, n);

        // Calcula os novos valores de X resultando na iteracao K+1
        computeNewCurrentX(currentX, previousX, normalizedA, normalizedB, n);
    }
    // A condicao de parada e alcancada quando o maior erro relativo encontrado na
    // iteracao K+1 for menor que a precisao estipulada como tolerancia
    while(getError(currentX, previousX, n) > ERROR_TOLERANCE);

    // Fim da execucao
    end = clock();

    // Calcula o tempo de CPU utilizado
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %fs for dimension %d\n", cpu_time_used, n);

    // Escreve os resultados no arquivo de saida
    fprintf(outputFile, "*** Sequential Results ***\n");
    showResults(A, currentX, B, n, outputFile);

    fclose(inputFile);
    fclose(outputFile);

    // Libera a memoria alocada
    cleanUp(&A, &currentX, &B, &normalizedA, &previousX, &normalizedB, n);
    return 0;
}

// Função para alocar matrizes e vetores
void initialize(float ***A, float **currentX, float **B, float ***normalizedA, float **previousX, float **normalizedB , int *n, FILE *file) {
    int i; // Variável utilizada para iteração
    fread(n, sizeof(int), 1, file);

    // Aloca espaco para a matriz A
    *A = (float **) malloc(*n * sizeof(float *));
    for(i = 0; i < *n; i++) {
        (*A)[i] = (float *) malloc(*n * sizeof(float));
    }

    // Aloca espaco para a matriz normalizada A
    *normalizedA = (float **) malloc(*n * sizeof(float *));
    for(i = 0; i < *n; i++) {
        (*normalizedA)[i] = (float *) malloc(*n * sizeof(float));
    }

    // Aloca espaco para os vetores x atual, x anterior,
    // B e B normalizado
    *currentX = (float *) malloc(*n * sizeof(float));
    *previousX = (float *) malloc(*n * sizeof(float));
    *B = (float *) malloc(*n * sizeof(float));
    *normalizedB = (float *) malloc(*n * sizeof(float));
}

// Retorna sempre o valor positivo de X
float absolute(float x) {
    return x < 0.0 ? -x : x;
}

// Le os dados do arquivo de entrada inputFile para preencher as estruturas A e B
void readDataFromInputFile(float **A, float *B, int n, FILE *inputFile) {
    // Iteradores
    int i;
    int j;

    // Preenche A
    for(i = 0; i < n; i ++) {
        for(j = 0; j < n; j++) {
            fread(&A[i][j], sizeof(float), 1, inputFile);
        }
    }

    // Preenche B
    for(i = 0; i < n; i ++) {
        fread(&B[i], sizeof(float), 1, inputFile);
    }
}

// Calcula os valores normalizados para a matriz A e para o vetor B.
// Os valores iniciais das variáveis X são definidos como o vetor B normalizado.
void normalize(float **A, float *currentX, float *B, float **normalizedA, float *normalizedB ,int n) {
    // Iteradores
    int i;
    int j;

    // Calcula a matriz (L* + R*) que tem a diagonal nula e os elementos
    // de uma linha divididos pelo elemento da diagonal de A
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

    // Calcula a matriz normalizada de B e o primeiro vetor X
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
void showResults(float **A, float *currentX, float *B, int n, FILE *outputFile) {
    int i;
    float calculatedResult = 0.0;

    int line = rand() % n;
    
    for(i = 0; i < n; i++) {
        fprintf(outputFile, "X[%d] = %f\n", i, currentX[i]);
    }
    
    fprintf(outputFile, "\nEquação aleatória para avaliação de corretude:\n");
    for (i = 0; i < n; i++) {
        fprintf(outputFile, "%2.3f * %2.3f", A[line][i], currentX[i]);
        calculatedResult += A[line][i] * currentX[i];
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
