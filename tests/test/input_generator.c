#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CONVERGE -1
#define MAXVAL 1024

void initialize(float ***A, float **B, int n);
void populate(float **A, float *B, int n);
float absolute(float x);
void generateLine(float **A, int line, int size);
void writeFile(float **A, float *B, int size, FILE *file);
void printAll(float **A, float *B, int n);

int main(int argc, char const *argv[]){
    FILE *output = NULL;
    float **A = NULL;
    float *B = NULL;
    int dimension;

    // Verifica se os argumentos necessários foram fornecidos
    if (argc != 3) {
        printf("usage: ./input_generator <dimension> <outputfile>\n");
        exit(0);
    }

    // Abre o arquivo de saída passado em argv[2]
    output = fopen(argv[2],"wb");
    if (output == NULL){
        fprintf(stderr, "Failed to create file [%s]\n", argv[2]);
        exit(1);
    }
    srand((unsigned)time(NULL));

    // Recupera o argumento com a dimensao da matriz
    dimension = atoi(argv[1]);

    // Inicializa duas estruturas A e B.
    // A matriz A tem dimensao dimension x dimension
    // A matriz B tem dimensao dimension
    initialize(&A, &B, dimension);

    // Preenche as estruturas A e B com valores aleatorios
    populate(A,B,dimension);

    // Escreve os dados gerados no arquivo de saida
    writeFile(A,B,dimension,output);

    //printAll(A,B,dimension);
    fclose(output);
    return 0;
}

void initialize(float ***A, float **B, int n) {
    int i;
    
    // Alocacao da matriz A
    *A = (float **) malloc(n * sizeof(float *));
    for(i = 0; i < n; i++) {
        (*A)[i] = (float *) malloc(n * sizeof(float));
    }
    
    // Alocacao do vetor B
    *B = (float *) malloc(n * sizeof(float));
}

void populate(float **A, float *B, int n){
    int i, j;

    // Gera n linhas estritamente diagonais
    for(i = 0; i < n; i ++) {
        generateLine(A, i, n);
    }

    // Preenche um vetor de dimensao N com numeros aleatorios
    for(i = 0; i < n; i ++) {
        B[i] = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
    }
}

void generateLine(float **A, int line, int size) {
    // Variavel armazena a maior entrada gerada pelo loop for
    float maxLineEntry = 0.0;

    // Variavel auxiliar
    float temp;

    // Contem a soma de todos os elementos exceto o de A[line][line]
    // que eh a diagonal da linha corrente
    float sum = 0.0;

    // Contem a diferenca entre sum e a diagonal
    float diff = 0.0;

    // Contem o indice da primeira ocorrencia de maxLineEntry
    int maxLineEntryIndex = 0;

    // Iterador
    int i;

    for (i = 0; i < size; ++i) {
        temp = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
        A[line][i] = temp;

        // Somente soma se o indice avaliado nao for a diagonal
        if (i != line){
            sum = sum + absolute(temp);
        }

        // Atualiza a variavel maxLineEntry
        if (absolute(temp) > absolute(maxLineEntry)){
            maxLineEntry = temp;
            maxLineEntryIndex = i;
        }
    }
    
    // Ate esse ponto, um vetor com numeros aleatorios foi gerado
    // A partir de agora as rotinas tornam essa linha estritamente
    // diagonal

    // Verdadeiro se a linha ja foi prrenchida corretamente com valores
    // que a fizeram destritamente diagonal
    if (sum < absolute(A[line][line])) {
        return;
    } else

    // Verifica se todos os elementos gerados foram nulos. Para tornar
    // a linha estritamente diagonal basta adicionar 1 a propria diagonal
    if (sum == 0){
        A[line][line] = 1;
    } else {
        // Entra nesse escopo no caso de os valores gerados nao forem nulos

        // Troca o valor da diagonal com o maxLineEntry
        temp = A[line][line];
        A[line][line] = maxLineEntry;
        A[line][maxLineEntryIndex] = temp;


        // Atualiza o valor de sum apos a troca da diagonal com maxLineEntry
        sum = sum - absolute(maxLineEntry) + absolute(temp);

        // Calcula quanto e necessario para a matriz se tornar estritamente
        // diagonal
        diff = sum - absolute(maxLineEntry);

        // Caso a soma dos elementos que nao pertencem a diagonal seja igual
        // ao valor da propria diagonal, basta adicionar 1 ou subtrair 1 da
        // propria diagonal e a matriz sera entao estritamente diagonal
        if (diff == 0.0) {
            if (A[line][line] > 0.0){
                A[line][line]++;
            } else {
                A[line][line]--;
            }
            return;
        } else {

            // Caso a soma dos elementos que nao pertencem a diagonal seja diferente
            // do valor da propria diagonal, e necessario somar ou subtrair algo de pelo
            // menos 1 unidade a mais de valor que diff

            // Gera um numero aleatorio que sera pelo menos diff + 1
            float correction = (float) (rand() % (int) diff + diff + 1);
            if (A[line][line] > 0.0){
                A[line][line] = A[line][line] + correction;
            } else {
                A[line][line] = A[line][line] - correction;
            }
        }
    }
}

// Retorna o valor sempre positivo de x
float absolute(float x) {
    return x < 0.0 ? -x : x;
}

void printAll(float **A, float *B, int n) {
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void writeFile(float **A, float *B, int size, FILE *file){

    FILE *output = NULL;
    int i;
    int j;

    //printf("Writing file...\n");
    fwrite(&size, sizeof(int), 1, file);

    for (i = 0 ; i < size ; i++) {
        for (j = 0 ; j < size ; j++) {
            fwrite(&A[i][j], sizeof(float), 1, file);
        }
    }

    for (i = 0 ; i < size ; i++) {
        fwrite(&B[i], sizeof(float), 1, file);
    }
}