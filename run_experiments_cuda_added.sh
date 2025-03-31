# Add the print_status function definition
print_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create necessary directories
mkdir -p results
mkdir -p debug_logs

# Create/clear the CUDA results file
CUDA_RESULTS_FILE="consolidated_results_cuda_added.csv"
echo "Version,Matrix,Execution_Time_ms,GFLOP_per_s,GB_per_s" > "$CUDA_RESULTS_FILE"

# Path for the combined results file (CPU + CUDA)
COMBINED_RESULTS_FILE="final_results.csv"

# Define matrices with their correct paths
MATRICES=(
    "comparison_matrices/D6-6/D6-6.mtx"
    "comparison_matrices/dictionary28/dictionary28.mtx"
    "comparison_matrices/Ga3As3H12.mtx"
    "comparison_matrices/bfly/bfly.mtx"
    "comparison_matrices/pkustk14.mtx"
    "comparison_matrices/roadNet-CA.mtx"
)

# Set up CUDA environment
print_status "Setting up CUDA environment..."
module load cuda

# Prepare nsight-compute temporary directory to avoid lock errors
UNITY_ID=$(whoami)
export TMPDIR=/tmp/nsight-compute-lock-${UNITY_ID}
mkdir -p $TMPDIR

# Function to extract execution time and performance metrics and add to CUDA results file
extract_and_append_cuda_result() {
    local version=$1
    local matrix_name=$2
    local result_file=$3
    
    # Use a more flexible approach - look for lines containing benchmarking info
    if grep -q "benchmarking CUDA COO-SpMV:" "$result_file"; then
        # Dump the exact line for debugging
        echo "Found benchmark line in $result_file:"
        grep "benchmarking CUDA COO-SpMV:" "$result_file"
        
        # Extract each value individually with correct field positions
        local exec_time=$(grep "benchmarking CUDA COO-SpMV:" "$result_file" | awk '{print $4}')
        local gflops=$(grep "benchmarking CUDA COO-SpMV:" "$result_file" | awk '{print $7}')
        local gbytes=$(grep "benchmarking CUDA COO-SpMV:" "$result_file" | awk '{print $9}')
                
        # Verify extracted values
        echo "Extracted: time=$exec_time, gflops=$gflops, gbytes=$gbytes"
        
        echo "$version,$matrix_name,$exec_time,$gflops,$gbytes" >> "$CUDA_RESULTS_FILE"
    else
        print_status "  WARNING: Could not find benchmark line in $result_file"
        # Dump file contents for debugging
        echo "File contents of $result_file:"
        cat "$result_file"
        
        # Try stderr log file as alternative
        local debug_file="debug_logs/cuda_${matrix_name}_debug.log"
        if grep -q "benchmarking CUDA COO-SpMV:" "$debug_file"; then
            print_status "  Found benchmark line in debug log file."
            
            # Extract each value individually with correct field positions
            local exec_time=$(grep "benchmarking CUDA COO-SpMV:" "$debug_file" | awk '{print $4}')
            local gflops=$(grep "benchmarking CUDA COO-SpMV:" "$debug_file" | awk '{print $7}')
            local gbytes=$(grep "benchmarking CUDA COO-SpMV:" "$debug_file" | awk '{print $9}')
            
            echo "Extracted from debug log: time=$exec_time, gflops=$gflops, gbytes=$gbytes"
            echo "$version,$matrix_name,$exec_time,$gflops,$gbytes" >> "$CUDA_RESULTS_FILE"
        else
            echo "$version,$matrix_name,0,0,0" >> "$CUDA_RESULTS_FILE"
        fi
    fi
}

# Function to extract MPI+CUDA results
extract_and_append_mpi_cuda_result() {
    local version=$1
    local matrix_name=$2
    local result_file=$3
    local num_gpus=$4  # Number of GPUs used
    
    # Use a more flexible approach - look for lines containing benchmarking info
    if grep -q "benchmarking MPI+CUDA COO-SpMV:" "$result_file"; then
        # Dump the exact line for debugging
        echo "Found benchmark line in $result_file:"
        grep "benchmarking MPI+CUDA COO-SpMV:" "$result_file"
        
        # Extract each value individually with correct field positions
        local exec_time=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$result_file" | awk '{print $4}')
        local gflops=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$result_file" | awk '{print $7}')
        local gbytes=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$result_file" | awk '{print $9}')
        
        # Verify extracted values
        echo "Extracted: time=$exec_time, gflops=$gflops, gbytes=$gbytes"
        
        echo "${version}-G${num_gpus},$matrix_name,$exec_time,$gflops,$gbytes" >> "$CUDA_RESULTS_FILE"
    else
        print_status "  WARNING: Could not find benchmark line in $result_file"
        # Dump file contents for debugging
        echo "File contents of $result_file:"
        cat "$result_file"
        
        # Try stderr log file as alternative
        local debug_file="debug_logs/mpi_cuda_np${version#*_np}_${matrix_name}_debug.log"
        if grep -q "benchmarking MPI+CUDA COO-SpMV:" "$debug_file"; then
            print_status "  Found benchmark line in debug log file."
            
            # Extract each value individually with correct field positions
            local exec_time=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$debug_file" | awk '{print $4}')
            local gflops=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$debug_file" | awk '{print $7}')
            local gbytes=$(grep "benchmarking MPI+CUDA COO-SpMV:" "$debug_file" | awk '{print $9}')
            
            echo "Extracted from debug log: time=$exec_time, gflops=$gflops, gbytes=$gbytes"
            echo "${version}-G${num_gpus},$matrix_name,$exec_time,$gflops,$gbytes" >> "$CUDA_RESULTS_FILE"
        else
            echo "${version}-G${num_gpus},$matrix_name,0,0,0" >> "$CUDA_RESULTS_FILE"
        fi
    fi
}

# Compile the CUDA and MPI-CUDA implementations
print_status "Compiling CUDA and MPI+CUDA SpMV implementations..."
make spmv-cuda spmv-mpi-cuda

# Define MPI process configurations to test
MPI_CONFIGS=(
    "2"   # 2 processes
    "4"   # 4 processes
    "8"   # 8 processes
    "16"  # 16 processes

)

# CUDA Standard run tests
print_status "Starting CUDA tests..."
for matrix in "${MATRICES[@]}"; do
    matrix_name=$(basename "$matrix" .mtx)
    print_status "Running CUDA on $matrix_name..."
    output_file="results/cuda_${matrix_name}.txt"
    # Capture both stdout and stderr
    ./spmv-cuda "$matrix" > "$output_file" 2> "debug_logs/cuda_${matrix_name}_debug.log"
    extract_and_append_cuda_result "CUDA" "$matrix_name" "$output_file"
    print_status "Completed CUDA on $matrix_name."
    

done

# Count available GPUs
print_status "Detecting available GPUs..."
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ "$num_gpus" -eq 0 ]; then
    print_status "WARNING: No GPUs detected, defaulting to 1"
    num_gpus=1
else
    print_status "Detected $num_gpus GPUs"
fi
# Run MPI-CUDA tests with different process counts
print_status "Starting MPI+CUDA tests..."
for np in "${MPI_CONFIGS[@]}"; do
    for matrix in "${MATRICES[@]}"; do
        matrix_name=$(basename "$matrix" .mtx)
        print_status "Running MPI+CUDA (np=$np, gpus=$num_gpus) on $matrix_name..."
        output_file="results/mpi_cuda_np${np}_${matrix_name}.txt"
        # Capture both stdout and stderr
        mpirun -np $np ./spmv-mpi-cuda "$matrix" > "$output_file" 2> "debug_logs/mpi_cuda_np${np}_${matrix_name}_debug.log"
        version="MPI+CUDA_np${np}"
        extract_and_append_mpi_cuda_result "$version" "$matrix_name" "$output_file" "$num_gpus"
        print_status "Completed MPI+CUDA (np=$np, gpus=$num_gpus) on $matrix_name."
    done
done

print_status "All tests completed. Results saved in $CUDA_RESULTS_FILE"

# Create a backup of raw CUDA results
cp "$CUDA_RESULTS_FILE" "${CUDA_RESULTS_FILE}.backup"

# Clean up any failed benchmarks (rows with zeros)
awk -F, 'NR==1 || ($3 != "0" && $3 != "")' "$CUDA_RESULTS_FILE" > "${CUDA_RESULTS_FILE}.clean"
mv "${CUDA_RESULTS_FILE}.clean" "$CUDA_RESULTS_FILE"

# Combine the original CPU results with CUDA results
if [ -f "consolidated_results.csv" ]; then
    print_status "Creating combined results file with CPU and CUDA data..."
    
    # Get the header from the original results file
    head -n 1 consolidated_results.csv > "$COMBINED_RESULTS_FILE"
    
    # Append all data from the original file (except header)
    tail -n +2 consolidated_results.csv >> "$COMBINED_RESULTS_FILE"
    
    # Process CUDA results to match the format of the original file
    print_status "Converting CUDA results to match CPU results format..."
    tail -n +2 "$CUDA_RESULTS_FILE" | while read -r line; do
        # Split the line into fields
        IFS=',' read -r version matrix exec_time gflops gbytes <<< "$line"
        
        # For CUDA versions, use "GPU" as configuration
        configuration="GPU"
        
        # Write to the combined file in the format: Version,Configuration,Matrix,Execution_Time_ms
        echo "$version,$configuration,$matrix,$exec_time" >> "$COMBINED_RESULTS_FILE"
    done
    
    print_status "Combined results saved to $COMBINED_RESULTS_FILE"
else
    print_status "No original consolidated_results.csv found. Creating new file with CUDA results only."
    echo "Version,Configuration,Matrix,Execution_Time_ms" > "$COMBINED_RESULTS_FILE"
    
    # Process CUDA results to match the format of the original file
    tail -n +2 "$CUDA_RESULTS_FILE" | while read -r line; do
        # Split the line into fields
        IFS=',' read -r version matrix exec_time gflops gbytes <<< "$line"
        
        # For CUDA versions, use "GPU" as configuration
        configuration="GPU"
        
        # Write to the combined file in the format: Version,Configuration,Matrix,Execution_Time_ms
        echo "$version,$configuration,$matrix,$exec_time" >> "$COMBINED_RESULTS_FILE"
    done
fi

# Generate a simple performance summary from CUDA results
print_status "Generating CUDA performance summary..."
echo -e "\nCUDA Performance Summary:" > cuda_performance_summary.txt
echo -e "Version\tMatrix\tExecution Time (ms)\tGFLOP/s\tGB/s" >> cuda_performance_summary.txt
awk -F, 'NR>1 {print $1"\t"$2"\t"$3"\t"$4"\t"$5}' "$CUDA_RESULTS_FILE" | sort >> cuda_performance_summary.txt
cat cuda_performance_summary.txt


# Install Python dependencies if visualization is requested
if [ -f "requirements.txt" ] && [ -f "plot_results_cuda_added.py" ]; then
    print_status "Installing Python dependencies for visualization..."
    pip3 install --user -r requirements.txt >/dev/null 2>&1 || {
        print_status "Warning: Could not install some Python dependencies. Visualization may fail."
    }
fi


# Copy the visualization script from the original location if it exists
if [ -f "plot_results_cuda_added.py" ]; then
    print_status "Running visualization script..."
    python3 plot_results_cuda_added.py
    print_status "Visualization completed. Image saved as perf-cmp-all.jpg"
else
    print_status "Visualization script not found. Skipping visualization."
fi

print_status "Job completed"