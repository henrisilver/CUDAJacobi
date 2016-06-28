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
    srand((unsigned)time(NULL));
    float **A = NULL;
    float *B = NULL;
    int dimension;
    if (argc != 3) {
        printf("usage: ./input_generator <dimension> <outputfile>\n");
        exit(0);
    }

    output = fopen(argv[2],"wb");
    if (output == NULL){
        fprintf(stderr, "Failed to create file [%s]\n", argv[2]);
        exit(1);
    }
    dimension = atoi(argv[1]);
    initialize(&A, &B, dimension);
    populate(A,B,dimension);
    writeFile(A,B,dimension,output);
    printAll(A,B,dimension);
    fclose(output);
    return 0;
}

void initialize(float ***A, float **B, int n) {
    int i;
    
    *A = (float **) malloc(n * sizeof(float *));
    for(i = 0; i < n; i++) {
        (*A)[i] = (float *) malloc(n * sizeof(float));
    }
    
    *B = (float *) malloc(n * sizeof(float));
}

void populate(float **A, float *B, int n){
    int i, j;

    for(i = 0; i < n; i ++) {
        generateLine(A, i, n);
    }

    for(i = 0; i < n; i ++) {
        B[i] = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
    }
}

void generateLine(float **A, int line, int size) {
    float maxLineEntry = 0.0;
    float temp;
    float sum = 0.0;
    float diff = 0.0;
    int maxLineEntryIndex = 0;
    int i;

    for (i = 0; i < size; ++i) {
        temp = (float) (rand() % (2 * MAXVAL + 1) - MAXVAL);
        A[line][i] = temp;
        if (i != line)
            sum = sum + absolute(temp);

        if (absolute(temp) > absolute(maxLineEntry)){
            maxLineEntry = temp;
            maxLineEntryIndex = i;
        }
    }
    
    if (sum < absolute(A[line][line])) {
        return;
    } else if (sum == 0){
        A[line][line] = 1;
    } else {
        temp = A[line][line];
        A[line][line] = maxLineEntry;
        A[line][maxLineEntryIndex] = temp;


        sum = sum - absolute(maxLineEntry) + absolute(temp);
        diff = sum - absolute(maxLineEntry);
        
        if (diff == 0.0) {
            if (A[line][line] > 0.0){
                A[line][line]++;
            } else {
                A[line][line]--;
            }
            return;
        } else {
            float correction = (float) (rand() % (int) diff / 2);
            if (A[line][line] > 0.0){
                A[line][line] = A[line][line] + correction;
            } else {
                A[line][line] = A[line][line] - correction;
            }
            
            /*while (diff >= 0.0) {
                while ((i = (int) (rand() % size)) == line);
                int mod = (int)A[line][i]+1;
                mod = mod == 0 ? 1 : mod;
                temp = (float)(rand() % mod);
                if (A[line][i] < 0.0) {
                    A[line][i] = A[line][i] + temp;
                } else {
                    A[line][i] = A[line][i] - temp;
                }
                diff = diff - temp;
            }*/
        }
    }
}

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

    printf("Writing file...\n");
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