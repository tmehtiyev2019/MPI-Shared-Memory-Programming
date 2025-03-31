#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for COO SpMV
__global__ void coo_spmv_kernel(int num_nonzeros, int start_idx, int end_idx, 
                               const int* rows, const int* cols, 
                               const float* vals, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = start_idx + idx;
    
    if (i < end_idx) {
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
    }
}

void usage(int argc, char** argv) {
    fprintf(stderr, "Usage: mpirun -n <processes> %s [my_matrix.mtx]\n", argv[0]);
    fprintf(stderr, "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n");
}

double benchmark_coo_spmv_mpi_cuda(coo_matrix *coo, float *x, float *y, int rank, int size) {
    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;
    int num_cols = coo->num_cols;
    
    // Calculate work distribution - each MPI process gets a chunk of nonzeros
    int nonzeros_per_proc = (num_nonzeros + size - 1) / size;
    int start_idx = rank * nonzeros_per_proc;
    int end_idx = min(start_idx + nonzeros_per_proc, num_nonzeros);
    int local_nonzeros = end_idx - start_idx;
    
    if (rank == 0) {
        printf("\nDistributing work: %d nonzeros across %d processes\n", num_nonzeros, size);
    }
    
    fprintf(stderr, "[Rank %d] Processing nonzeros %d to %d (%d elements)\n", 
            rank, start_idx, end_idx - 1, local_nonzeros);

    // Allocate device memory
    int* d_rows;
    int* d_cols;
    float* d_vals;
    float* d_x;
    float* d_y;
    
    CUDA_CHECK(cudaMalloc((void**)&d_rows, num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cols, num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_vals, num_nonzeros * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));
    
    // Copy all data to device - in a more optimized version, we could transfer only the needed chunk
    CUDA_CHECK(cudaMemcpy(d_rows, coo->rows, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, coo->cols, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, coo->vals, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (local_nonzeros + blockSize - 1) / blockSize;
    
    // Local result vector for each process
    float* local_y = (float*)calloc(num_rows, sizeof(float));
    
    // Warmup
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    // Initialize device y to zeros
    CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));
    
    // Launch kernel for warmup - each process processes its chunk
    coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, start_idx, end_idx, 
                                            d_rows, d_cols, d_vals, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double estimated_time = seconds_elapsed(&time_one_iteration);
    
    // Determine number of iterations
    int num_iterations = MAX_ITER;
    
    if (rank == 0) {
        printf("\tPerforming %d iterations\n", num_iterations);
    }
    
    // Time several SpMV iterations
    timer t;
    timer_start(&t);
    
    // // Initialize y to zeros for benchmarking
    // CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));
    
    for (int j = 0; j < num_iterations; j++) {
        // Each process processes its chunk on GPU
        coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, start_idx, end_idx, 
                                                d_rows, d_cols, d_vals, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(local_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Global sum across all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate memory for the global result on rank 0
    float* global_y = NULL;
    if (rank == 0) {
        global_y = (float*)malloc(num_rows * sizeof(float));
    }
    
    // Reduce the local results to global result on rank 0
    MPI_Reduce(local_y, global_y, num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Broadcast the result to all processes
    if (rank == 0) {
        memcpy(y, global_y, num_rows * sizeof(float));
        free(global_y);
    }
    MPI_Bcast(y, num_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Calculate performance metrics
    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    
    if (rank == 0) {
        double sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : 
                        (2.0 * (double)num_nonzeros / sec_per_iteration) / 1e9;
        
        // Calculate bandwidth usage (bytes read + bytes written per iteration)
        size_t bytes_read = (num_nonzeros * sizeof(int) * 2) + // rows and cols arrays
                            (num_nonzeros * sizeof(float)) +   // vals array
                            (num_cols * sizeof(float));        // x vector
        size_t bytes_written = num_rows * sizeof(float);       // y vector
        
        double GBYTEs = (sec_per_iteration == 0) ? 0 : 
                        ((double)(bytes_read + bytes_written) / sec_per_iteration) / 1e9;
        
        printf("\tbenchmarking MPI+CUDA COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
               msec_per_iteration, GFLOPs, GBYTEs);
    }
    
    // Clean up
    free(local_y);
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return msec_per_iteration;
}

int main(int argc, char** argv) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        usage(argc, argv);
    }
    
    // Check arguments
    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: No matrix file specified.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    char *mm_filename = argv[1];
    coo_matrix coo;
    
    // Only rank 0 reads the matrix
    if (rank == 0) {
        read_coo_matrix(&coo, mm_filename);
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", 
               mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        
        // Fill matrix with original values (uncomment below to use random values instead)
        
        srand(13);
        for(int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
            // coo.vals[i] = (float)((i % 9) + 1);
        }

        // Add debug print:
        printf("First few matrix values: %f %f %f %f\n", 
            coo.vals[0], coo.vals[1], coo.vals[2], coo.vals[3]);
        
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
        
        if (!coo.rows || !coo.cols || !coo.vals) {
            fprintf(stderr, "[Rank %d] Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Broadcast matrix data
    MPI_Bcast(coo.rows, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.cols, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Allocate and initialize vectors
    float *x = (float*)malloc(coo.num_cols * sizeof(float));
    float *y = (float*)calloc(coo.num_rows, sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "[Rank %d] Vector memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize x with all 1's for easier verification
    for (int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0);
        // x[i] = 1.0;
    }
    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Get number of available CUDA devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Set CUDA device based on rank, but make sure we don't exceed available devices
    int device = rank % deviceCount;
    fprintf(stderr, "[Rank %d] Using CUDA device %d (of %d available)\n", rank, device, deviceCount);
    CUDA_CHECK(cudaSetDevice(device));
    
    // Run the benchmark
    benchmark_coo_spmv_mpi_cuda(&coo, x, y, rank, size);
    
    // Rank 0 prints results
    if (rank == 0) {
        printf("Result vector y:\n");
        for (int i = 0; i < 5 && i < coo.num_rows; i++) {
            fprintf(stderr, "y[%d] = %f\n", i, y[i]);
        }
    }
    
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
    
    MPI_Finalize();
    return 0;
}