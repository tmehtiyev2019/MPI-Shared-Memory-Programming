#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
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

void usage(int argc, char** argv) {
    fprintf(stderr, "Usage: mpirun -n 8 %s [my_matrix.mtx]\n", argv[0]);
    fprintf(stderr, "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n");
}


double benchmark_coo_spmv(coo_matrix *coo, float *x, float *y, int rank, int size) {
    int num_nonzeros = coo->num_nonzeros;
    
    // Calculate work distribution
    int nonzeros_per_proc = (num_nonzeros + size - 1) / size;
    int start_idx = rank * nonzeros_per_proc;
    int end_idx = start_idx + nonzeros_per_proc;
    
    // Boundary checks
    if (rank == size - 1 || end_idx > num_nonzeros) {
        end_idx = num_nonzeros;
    }
    
    fprintf(stderr, "[Rank %d] Work range: %d to %d (total nonzeros: %d)\n", 
            rank, start_idx, end_idx, num_nonzeros);

    // Local array for accumulation across all iterations
    float *local_y = (float*)calloc(coo->num_rows, sizeof(float));
    float *global_y = (rank == 0) ? (float*)malloc(coo->num_rows * sizeof(float)) : NULL;
    
    if (!local_y || (rank == 0 && !global_y)) {
        fprintf(stderr, "[Rank %d] Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Warmup iteration (only for timing)
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    for (int i = start_idx; i < end_idx; i++) {
        local_y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
    
    double estimated_time = seconds_elapsed(&time_one_iteration);

    // Calculate iterations like sequential version
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0) {
        num_iterations = MAX_ITER;
    } else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
    }

    if (rank == 0) {
        fprintf(stderr, "\tPerforming %d iterations\n", num_iterations);
    }


    // Main timing loop - do all iterations locally first
    timer t;
    timer_start(&t);
    
    // Do all iterations locally
    for (int iter = 0; iter < num_iterations; iter++) {
        for (int i = start_idx; i < end_idx; i++) {
            local_y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
        fprintf(stderr, "[Rank %d] Completed local iteration %d\n", rank, iter);
    }

    fprintf(stderr, "[Rank %d] All local iterations complete. Starting global reduction.\n", rank);

    // Single reduction for final result
    MPI_Reduce(local_y, global_y, coo->num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    


    
    // All processes complete their iterations
    MPI_Barrier(MPI_COMM_WORLD);

    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;

        // Single broadcast of final result
        if (rank == 0) {
            memcpy(y, global_y, coo->num_rows * sizeof(float));
        }
        MPI_Bcast(y, coo->num_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : 
                       (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
        double GBYTEs = (sec_per_iteration == 0) ? 0 : 
                       ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;

        printf("Total Execution Time: %f ms per iteration\n", msec_per_iteration);
        printf("Performance: %f GFLOP/s, %f GB/s\n", GFLOPs, GBYTEs);
        fprintf(stderr, "\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
                msec_per_iteration, GFLOPs, GBYTEs);
    }

    free(local_y);
    if (rank == 0 && global_y) {
        free(global_y);
    }

    return msec_per_iteration;
}


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    usage(argc, argv);

    char *mm_filename = argv[1];
    coo_matrix coo;

    // Only rank 0 reads the matrix
    if (rank == 0) {
        read_coo_matrix(&coo, mm_filename);
        fprintf(stderr, "[Rank 0] Finished reading matrix: %s (Rows: %d, Cols: %d, Nonzeros: %d)\n",
                mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult, especially in single precision
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
        // coo.vals[i] = 1.0;
    }
    }

    // Broadcast matrix metadata
    MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_nonzeros, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for other ranks before receiving data
    if (rank != 0) {
        coo.rows = (int*)malloc(coo.num_nonzeros * sizeof(int));
        coo.cols = (int*)malloc(coo.num_nonzeros * sizeof(int));
        coo.vals = (float*)malloc(coo.num_nonzeros * sizeof(float));
    }

    // Broadcast matrix data
    MPI_Bcast(coo.rows, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.cols, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Allocate and initialize vectors
    float *x = (float*)malloc(coo.num_cols * sizeof(float));
    float *y = (float*)malloc(coo.num_rows * sizeof(float));

    if (rank == 0) {
        srand(13);
        for (int i = 0; i < coo.num_cols; i++) {
            x[i] = (float)rand() / RAND_MAX;
        }
    }

    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < coo.num_rows; i++)
    y[i] = 0;
    memset(y, 0, coo.num_rows * sizeof(float));

    // Perform SpMV benchmark
    benchmark_coo_spmv(&coo, x, y, rank, size);

    // Rank 0 prints results
    if (rank == 0) {
        fprintf(stderr, "[Rank 0] Final y vector (first 5 elements):\n");
        for (int i = 0; i < 5 && i < coo.num_rows; i++) {
            fprintf(stderr, "y[%d] = %f\n", i, y[i]);
        }
    }
    fprintf(stderr, "[Rank %d] Waiting at final barrier...\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "[Rank %d] Passed final barrier, proceeding to cleanup.\n", rank);


    // Clean up
    free(x);
    free(y);
    if (rank != 0) {
        free(coo.rows);
        free(coo.cols);
        free(coo.vals);
    } else {
        delete_coo_matrix(&coo);
    }
    fprintf(stderr, "[Rank %d] Going to MPI_Finalize.\n", rank);
    MPI_Finalize();
    fprintf(stderr, "[Rank %d] I am done.\n", rank);
    return 0;
}