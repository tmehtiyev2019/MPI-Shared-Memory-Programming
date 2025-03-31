#include <stdio.h>
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
__global__ void coo_spmv_kernel(int num_nonzeros, const int* rows, const int* cols, const float* vals, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_nonzeros) {
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
    }
}

void usage(int argc, char** argv) {
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

double benchmark_coo_spmv_cuda(coo_matrix* coo, float* x, float* y) {
    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;
    int num_cols = coo->num_cols;
    
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
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_rows, coo->rows, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, coo->cols, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, coo->vals, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (num_nonzeros + blockSize - 1) / blockSize;
    
    // Warmup
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    // Initialize y to zeros
    CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));
    
    // Launch kernel for warmup
    coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double estimated_time = seconds_elapsed(&time_one_iteration);
    
    // Determine number of iterations
    int num_iterations = MAX_ITER;
    printf("\tPerforming %d iterations\n", num_iterations);
    
    // Time several SpMV iterations
    timer t;
    timer_start(&t);
    
    for (int j = 0; j < num_iterations; j++) {
        // Reset y to zeros for each iteration
        // CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));
        
        // Launch kernel
        coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    
    // Calculate bandwidth usage (bytes read + bytes written per iteration)
    size_t bytes_read = (num_nonzeros * sizeof(int) * 2) + // rows and cols arrays
                        (num_nonzeros * sizeof(float)) +   // vals array
                        (num_cols * sizeof(float));        // x vector (assuming all elements accessed)
    size_t bytes_written = num_rows * sizeof(float);       // y vector
    
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)(bytes_read + bytes_written) / sec_per_iteration) / 1e9;
    
    printf("\tbenchmarking CUDA COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
           msec_per_iteration, GFLOPs, GBYTEs);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return msec_per_iteration;
}

int main(int argc, char** argv) {
    if (get_arg(argc, argv, "help") != NULL) {
        usage(argc, argv);
        return 0;
    }

    char* mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else {
        mm_filename = argv[1];
    }

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // Fill matrix with random values
    srand(13);
    for (int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
        // coo.vals[i] = (float)((i % 9) + 1);
    }

    // Add debug print:
    printf("First few matrix values: %f %f %f %f\n", 
    coo.vals[0], coo.vals[1], coo.vals[2], coo.vals[3]);
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

#ifdef TESTING
    // Print in COO format
    printf("Writing matrix in COO format to test_COO ...");
    FILE* fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif

    // Initialize host arrays
    float* x = (float*)malloc(coo.num_cols * sizeof(float));
    float* y = (float*)malloc(coo.num_rows * sizeof(float));

    for (int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0);
        // x[i] = 1.0;
    }
    for (int i = 0; i < coo.num_rows; i++) {
        y[i] = 0;
    }

    /* Benchmarking */
    double cuda_msec = benchmark_coo_spmv_cuda(&coo, x, y);

    // Print result vector y to stdout
    printf("Result vector y:\n");
    for (int i = 0; i < 5 && i < coo.num_rows; i++) {
        fprintf(stderr, "y[%d] = %f\n", i, y[i]);
    }

#ifdef TESTING
    printf("Writing x and y vectors ...");
    FILE* fp = fopen("test_x", "w");
    for (int i = 0; i < coo.num_cols; i++) {
        fprintf(fp, "%f\n", x[i]);
    }
    fclose(fp);
    fp = fopen("test_y", "w");
    for (int i = 0; i < coo.num_rows; i++) {
        fprintf(fp, "%f\n", y[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    return 0;
}