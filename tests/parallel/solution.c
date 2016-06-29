#include "solution.h"
#include <omp.h>


// Macros for accessing a specific section of the image (before and after
// processing it). Each thread is responsible for a certain amount of
// sections.
#define image_section(x,y,i) pixel_section[i][x * width + y]
#define smooth_section(x,y,i) filtered_section[i][x * width + y]

// When we are unable to evenly divide the number of images to be processed
// among threads, some threads will be responsible for processing one more
// image than other threads. For example, if we hava 8 threads and 10 images,
// each of the 8 threads will have to process at least one image. Also, 2 of
// those threads will need to process one extra image, as 10 % 8 = 2.
// To implement that, we use the following macros:
//
// extra_process_seek is responsible for setting the offset in the file
// descriptor where each thread will start processing the images. So, considering
// the example above, of 10 imges and 8 threads, thread 0 will start at position 0,
// thread 1 will staret at position 2 (because thread 0 will deal with two images),
// thread 2 will start at position 4 (thread 2 will also handle two images), thread
// 3 will start at position 5 and so on. extra_process_seek defines the offset used
// by each thread to read the input file, based on the number of additional images
// each of the previous threads are dealing with.
//
// extra_process_iteration is responsible for defining whether a thread must execute
// an extra iteration to process an additional image. So, in the example of 8 thread
// and 10 images, threads 0 and 1 will need 1 extra iteration each, as they are the
// only threads handling more than 1 image.
#define extra_process_seek(current_process_number, number_of_images) (current_process_number < (number_of_images % omp_get_max_threads()) ? current_process_number : number_of_images % omp_get_max_threads())
#define extra_process_iteration(current_process_number, number_of_images) (current_process_number < (number_of_images % omp_get_max_threads()) ? 1 : 0)

int solution(int argc, char** argv)
{
    
    // Input and output files to be used for reading the
    // images' height and width and calculate the number of images.
    FILE *in;
    FILE *out;
    
    // Opening input file
    in = fopen(argv[1], "rb");
    if (in == NULL) {
        printf("Unable to open file: %s\n", argv[1]);
        return 1;
    }
    
    // Opening output file
    out = fopen(argv[2], "wb+");
    if (out == NULL) {
        printf("Unable to open file: %s\n", argv[2]);
        return 1;
    }
    
    // Variables responsible for calculating the total file size,
    // the number of images in the file, the images' height and width.
    long int size, number_of_images;
    int width, height;
    
    fread(&width, sizeof(width), 1, in);
    fread(&height, sizeof(height), 1, in);
    
    fwrite(&width, sizeof(width), 1, out);
    fwrite(&height, sizeof(height), 1, out);
    
    // Finding the total number of images usgin the file size:
    
    // Get file size
    fseek(in, 0, SEEK_END);
    size = ftell(in);
    
    // Disconsider the initial bytes used to represent the images' width
    // and height
    size = size - sizeof(height) - sizeof(width);
    
    // Number of images is the total size of all images divided by
    // the size of one image
    number_of_images = size / (height * width * sizeof(int));

    fclose(in);
    fclose(out);

    // Arrays used to move around a certain element in a pixel matrix,
    // in order to get all the neighbors of a given pixel
    int DY[] = { -1, -1, -1, +0, +0, +0, +1, +1, +1 };
    int DX[] = { -1, +0, +1, -1, +0, +1, -1, +0, +1 };
   
    // x and y are used to iterate over the pixel matrix (y for height, x
    // for width). d is used to select each of the 9 possible neighbors
    // of a given pixel, which are accessed using dx and dy
    int x, y, d, dx, dy;
    
    // i representes the current process number
    // j represents the number of iterations each thread will perform
    int i,j;

    // File descriptors to be used for input/output for each thread
    FILE* in_section[omp_get_max_threads()];
    FILE* out_section[omp_get_max_threads()];
    
    // Pixels matrices for each thread
    int* pixel_section[omp_get_max_threads()];
    int* filtered_section[omp_get_max_threads()];

    // Extra offset needed to read each of the file's sections
    long int seek_offset;

    // Sequential loop. In this part, the arrays of file descriptors and
    // the pixel matrices for each thread are set up
    for (i = 0; i < omp_get_max_threads(); i++)
    {
        // This if deals with the case when we have more threads than images.
        if(i < number_of_images) {
            // Allocate memory for each pixel matrix and each processed pixel matrix
            pixel_section[i] = (int *) malloc(height * width * sizeof(int));
            filtered_section[i] = (int *) malloc(height * width * sizeof(int));

            // Opens the input file for each thread
            in_section[i] = fopen(argv[1], "rb");
            if (in_section[i] == NULL) {
                printf("Unable to open file: %s\n", argv[1]);
                return 1;
            }

            // If there are more threads than images, the seek_offset corresponds
            // simply to the number of the thread
            if(number_of_images < omp_get_max_threads()) {
                seek_offset = i * height * width * sizeof(int) + 2 * sizeof(int);
            } else {
                // The seek offset is determined by:
                // - the current thread number, i
                // - the default number of images that each thread must process, number_of_images/omp_get_max_threads()
                // - the additional offset that may be needed if the number of images
                //   cannot be divided evenly by the number of threads, calculated by extra_process_seek()
                // - the height and the width of each image, and the size of the integer used to represent pixels
                // - the initial two integers in the file, which define the height and width of the images
                seek_offset = (i*((number_of_images/omp_get_max_threads()))+extra_process_seek(i,number_of_images)) * height * width * sizeof(int) + 2 * sizeof(int);
            }


            // And moves the file descriptor to the first image the thread must handle,
            // determined by the seek offset
            fseek(in_section[i], seek_offset, SEEK_SET);
            
            // The same happens for the output file
            out_section[i] = fopen(argv[2], "rb+");
            if (out_section[i] == NULL) {
                printf("Unable to open file: %s\n", argv[2]);
                return 1;
            }

            fseek(out_section[i], seek_offset, SEEK_SET);
        }
    }

    // Number of iterations per thread.
    int number_of_iterations;

    // number of neighbors per pixel
    int neighbor_count;

    // Parallel for loop used so that each thread may process the images assigned to them concurrently.
    // Each thread reads part of the input file from their own file descriptor, storing the pixels in
    // their own pixel matrix. Then, each thread processes the pixel matrices assigned to them and
    // store the results in the filtered matrices. In the end, the content of those filtered matrices
    // is stored in the section of the output file that was assgigned to each of the threads.
    #pragma omp parallel for private(dx, dy, x, y, d, j, number_of_iterations, neighbor_count)
    // For each thread
    for(i = 0; i< omp_get_max_threads(); i++) {

        // This if deals with the case when we have more threads than images.
        if(i < number_of_images) {

            // If there are more threads than images, the number_of_images is only 1.
            if(number_of_images < omp_get_max_threads()) {
               number_of_iterations = 1;
            } else {
                // Each thread will have a custom number of iterations, as some of them may process 1 extra
                // image.
                // - number_of_images/omp_get_max_threads() is the default number of iterations.
                // - extra_process_iteration() calculates whether or not an extra iteration is needed for the current thread, i.
                number_of_iterations = ( number_of_images/omp_get_max_threads() + extra_process_iteration(i,number_of_images));
            }

            // We iterate number_of_iterations times
            for(j = 0; j < number_of_iterations; j++) {
                
                // And then we perform the Smooth process for each image.
                // 1 - the image is read.
                // 2 - The Smooth operation is performed, considering all possible neighbors of a given pixel
                int result = fread(pixel_section[i], height * width * sizeof(int), 1, in_section[i]);

                if (result){
                    for (y = 0; y < height; y++) {
                        for (x = 0; x < width; x++) {
                            long long int sum = 0;
                            neighbor_count = 0;
                            for (d = 0; d < 9; d++) {
                                dx = x + DX[d];
                                dy = y + DY[d];
                                if (dx >= 0 && dx < width && dy >= 0 && dy < height) {
                                    sum += image_section(dy, dx, i);
                                    neighbor_count++;
                                }
                            }
                            smooth_section(y, x, i) = sum / neighbor_count;
                        }
                    }
                    fwrite(filtered_section[i], height * width * sizeof(int), 1, out_section[i]);
                }
            }
        }
    }

    // Memory is deallocated and files are closed.
    for(i = 0; i< omp_get_max_threads(); i++) {

        // This if deals with the case when we have more threads than images.
        if(i < number_of_images) {
            free(pixel_section[i]);
            free(filtered_section[i]);
            fclose(in_section[i]);
            fclose(out_section[i]);
        }
    }

    return 0;
}
