#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "../common/utils.h"

FILE* openFile(char* filename, char* extension, char* mode)
{
    FILE* filePointer;
    char fullname[50];

    strcpy(fullname, filename);
    strcat(fullname, extension);

    filePointer = fopen(fullname, mode);

    return filePointer;
}

int fileOpenError(FILE* filePointer)
{
    if (filePointer == NULL)
    {
        printf("Unable to open file\n");
        return 1;
    }
    else
    {
        return 0;
    }
}

int fileInfo (char* filename, int* width, int* height, int* frames, long int* size)
{
    FILE *in;
    char _filename[256];
    long int start, end, area;

    strcpy(_filename, "../");
    strcat(_filename, filename);

    in = openFile(_filename, "", "rb");
    if (fileOpenError(in)) return 2;

    // Reading initial file information and copying that to output file

    fread(width, sizeof(*width), 1, in);
    fread(height, sizeof(*height), 1, in);

    area = *width * (*height);

    // Gathering information to compute frame area
    start = ftell(in);
    fseek(in, 0, SEEK_END);
    end = ftell(in);
    fseek(in, start, SEEK_SET);

    *size = end - start;
    *frames = (end - start) / (area * sizeof(int));

    fclose(in);
    return 0;
}

int metricWriter(char* filename, char* testCaseName, double executionTime)
{
    FILE* out;

    out = openFile(filename, "", "a+");

    if (fileOpenError(out))
    {
        return 1;
    }

    fprintf(out, "%s %lf\n", testCaseName, executionTime);

    fclose(out);
    return 0;
}

int metricReader(char* filename, int* size, char** tests, double* results)
{
    FILE* in;
    char testCaseName[50];
    double result;
    int n = 0;

    in = openFile(filename, "", "r");
    if (fileOpenError(in))
    {
        return 1;
    }

    while(!feof(in))
    {
        fscanf(in, "%s %lf\n", testCaseName, &result);

        //printf("Read %s %lf\n", testCaseName, result);
        strcpy(tests[n], testCaseName);
        results[n] = result;
        n++;
    }

    *size = n;
    fclose(in);
}

int main(int argc, char* argv[])
{
    char **testsA, **testsB;
    double resultsA[50], resultsB[50];
    int n, m;
    FILE* out;
    int width, height, frames;
    long int filesize;

    int i, j;

    if (argc < 4)
    {
        printf("Usage: ./%s sequential_results parallel_results metric_stats\n", argv[0]);
        return 1;
    }

    testsA = (char**) malloc(50 * sizeof(char**));
    for (i = 0; i < 50; i++) testsA[i] = (char*) malloc(50 * sizeof(char));

    metricReader(argv[1], &n, (char**) testsA, (double*) resultsA);

    testsB = (char**) malloc(50 * sizeof(char**));
    for (i = 0; i < 50; i++) testsB[i] = (char*) malloc(50 * sizeof(char));

    metricReader(argv[2], &m, (char**) testsB, (double*) resultsB);

    out = openFile(argv[3], "", "w+");
    if (fileOpenError(out))
    {
        printf("Error traceback: file %s\n", argv[3]);
        return 1;
    }

    fprintf(out, "Overall pairwise metrics results\n\n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            if (strcmp(testsA[i], testsB[j]) == 0)
            {
                fprintf(out, "For test case %s:\n", testsA[i]);

                if(fileInfo(testsA[i], &width, &height, &frames, &filesize) == 0)
                {
                    fprintf(out, "\tWidth: %d Height: %d Frames: %d File size: %ld\n", width, height, frames, filesize);
                }

                fprintf(out, "\tSequential time: %lf\n\tParallel time: %lf\n\tSpeedup: %lf\n\n", resultsA[i], resultsB[j], resultsA[i] / resultsB[j]);
            }
        }
    }

    fclose(out);

    for (i = 0; i < 50; i++) free(testsA[i]);
    free(testsA);

    for (i = 0; i < 50; i++) free(testsB[i]);
    free(testsB);

    return 0;
}
