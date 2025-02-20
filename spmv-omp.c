#include <stdio.h>
#include <omp.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    fprintf(stderr,"Usage: %s [my_matrix.mtx]\n", argv[0]);
    fprintf(stderr,"Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    int num_nonzeros = coo->num_nonzeros;

    // warmup with OpenMP
    timer time_one_iteration;
    timer_start(&time_one_iteration);

    #pragma omp parallel for
    for (int i = 0; i < num_nonzeros; i++){   
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    double estimated_time = seconds_elapsed(&time_one_iteration); 

    int num_iterations = MAX_ITER;

    if (estimated_time == 0) {
        num_iterations = MAX_ITER;
    } else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    }



    fprintf(stderr,"\tPerforming %d iterations\n", num_iterations);

    // time several SpMV iterations with OpenMP
    timer t;
    timer_start(&t);

    for(int j = 0; j < num_iterations; j++) {
        // Main SpMV computation
        #pragma omp parallel for
        for (int i = 0; i < num_nonzeros; i++){   
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }

    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;

    printf("Total Execution Time: %f ms per iteration\n", msec_per_iteration);
    printf("Performance: %f GFLOP/s, %f GB/s\n", GFLOPs, GBYTEs);

    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        fprintf(stderr,"Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
    }
    
    fprintf(stderr,"\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    //initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));

    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }
    for(int i = 0; i < coo.num_rows; i++)
        y[i] = 0;
    memset(y, 0, coo.num_rows * sizeof(float));

    /* Benchmarking */
    double coo_gflops;
    coo_gflops = benchmark_coo_spmv(&coo, x, y);

    // Print result vector y to stdout
    fprintf(stderr,"Result vector y (first 5 elements):\n");
    for(int i = 0; i < 5 && i < coo.num_rows; i++) {
        fprintf(stderr,"y[%d] = %f\n", i, y[i]);
    }

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    return 0;
}