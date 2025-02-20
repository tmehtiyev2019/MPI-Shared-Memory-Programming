#!/bin/bash

# Add the print_status function definition
print_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

print_status "Setting up Python environment..."
pip3 install --user -r requirements.txt


# Create necessary directories
mkdir -p results
mkdir -p debug_logs

# Create/clear the consolidated results file
RESULTS_FILE="consolidated_results.csv"
echo "Version,Configuration,Matrix,Execution_Time_ms" > "$RESULTS_FILE"

# Define matrices with their correct paths
MATRICES=(
    "comparison_matrices/D6-6/D6-6.mtx"
    "comparison_matrices/dictionary28/dictionary28.mtx"
    "comparison_matrices/Ga3As3H12.mtx"
    "comparison_matrices/bfly/bfly.mtx"
    "comparison_matrices/pkustk14.mtx"
    "comparison_matrices/roadNet-CA.mtx"
)

# Function to extract execution time and add to consolidated file
extract_and_append_result() {
    local version=$1
    local config=$2
    local matrix_name=$3
    local result_file=$4
    
    # Extract execution time (assuming it's in the format "Total Execution Time: X.XXX ms per iteration")
    local exec_time=$(grep "Total Execution Time:" "$result_file" | awk '{print $4}')
    echo "$version,$config,$matrix_name,$exec_time" >> "$RESULTS_FILE"
}


# Sequential version
print_status "Starting Sequential tests..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running Sequential on $matrix_name..."
    output_file="results/seq_${matrix_name}.txt"
    ./spmv "$matrix" > "$output_file" 2> "debug_logs/seq_${matrix_name}_debug.log"
    extract_and_append_result "Sequential" "N/A" "$matrix_name" "$output_file"
    print_status "Completed Sequential on $matrix_name."
done
print_status "Sequential tests finished."

# MPI version (-N 8 -n 8)
print_status "Starting MPI tests..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running MPI on $matrix_name..."
    output_file="results/mpi_${matrix_name}.txt"
    mpirun -np 8 ./spmv-mpi "$matrix" > "$output_file" 2> "debug_logs/mpi_${matrix_name}_debug.log"
    extract_and_append_result "MPI" "N8_n8" "$matrix_name" "$output_file"
    print_status "Completed MPI on $matrix_name."
done
print_status "MPI tests finished."

# OpenMP version (-N 1 -n 8)
export OMP_NUM_THREADS=8
print_status "Starting OpenMP tests..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running OpenMP on $matrix_name..."
    output_file="results/omp_${matrix_name}.txt"
    ./spmv-omp "$matrix" > "$output_file" 2> "debug_logs/omp_${matrix_name}_debug.log"
    extract_and_append_result "OpenMP" "N1_n8" "$matrix_name" "$output_file"
    print_status "Completed OpenMP on $matrix_name."
done
print_status "OpenMP tests finished."

# Hybrid versions
export MV2_ENABLE_AFFINITY=0

# Hybrid N8 n8
export OMP_NUM_THREADS=1
print_status "Starting Hybrid tests (N8 n8)..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running Hybrid (N8 n8) on $matrix_name..."
    output_file="results/hybrid_8n_${matrix_name}.txt"
    mpirun -np 8 ./spmv-hybrid "$matrix" > "$output_file" 2> "debug_logs/hybrid_8n_${matrix_name}_debug.log"
    extract_and_append_result "Hybrid" "N8_n8" "$matrix_name" "$output_file"
    print_status "Completed Hybrid (N8 n8) on $matrix_name."
done
print_status "Hybrid (N8 n8) tests finished."

# Hybrid N4 n8
export OMP_NUM_THREADS=2
print_status "Starting Hybrid tests (N4 n8)..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running Hybrid (N4 n8) on $matrix_name..."
    output_file="results/hybrid_4n_${matrix_name}.txt"
    mpirun -np 4 ./spmv-hybrid "$matrix" > "$output_file" 2> "debug_logs/hybrid_4n_${matrix_name}_debug.log"
    extract_and_append_result "Hybrid" "N4_n8" "$matrix_name" "$output_file"
    print_status "Completed Hybrid (N4 n8) on $matrix_name."
done
print_status "Hybrid (N4 n8) tests finished."

# Hybrid N2 n8
export OMP_NUM_THREADS=4
print_status "Starting Hybrid tests (N2 n8)..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running Hybrid (N2 n8) on $matrix_name..."
    output_file="results/hybrid_2n_${matrix_name}.txt"
    mpirun -np 2 ./spmv-hybrid "$matrix" > "$output_file" 2> "debug_logs/hybrid_2n_${matrix_name}_debug.log"
    extract_and_append_result "Hybrid" "N2_n8" "$matrix_name" "$output_file"
    print_status "Completed Hybrid (N2 n8) on $matrix_name."
done
print_status "Hybrid (N2 n8) tests finished."

print_status "All tests completed. Results consolidated in $RESULTS_FILE"

# Generate plot if python and required packages are available
if command -v python3 &>/dev/null; then
    print_status "Generating performance plot..."
    python3 plot_results.py
    print_status "Plot generated as perf-cmp.jpg"
fi

print_status "Job completed"