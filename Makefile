# Compiler settings
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Flags
CFLAGS = -O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAGS = -O3 -lm
OMPFLAGS = -fopenmp
NVFLAGS = -O3 -std=c++11 -I./include/ -diag-suppress 177,2464

# All targets
all: spmv spmv-mpi spmv-omp spmv-hybrid spmv-cuda spmv-mpi-cuda

# Object files
OBJS = mmio.o
CPU_OBJS = $(OBJS) spmv.o
MPI_OBJS = $(OBJS) spmv-mpi.o
OMP_OBJS = $(OBJS) spmv-omp.o
HYBRID_OBJS = $(OBJS) spmv-hybrid.o

# Pattern rules
%.o: %.c
	$(CC) -o $@ -c $(CFLAGS) $<

# Special rules for each version
spmv-mpi.o: spmv-mpi.c
	$(MPICC) -o $@ -c $(CFLAGS) $<

spmv-omp.o: spmv-omp.c
	$(CC) -o $@ -c $(CFLAGS) $(OMPFLAGS) $<

spmv-hybrid.o: spmv-hybrid.c
	$(MPICC) -o $@ -c $(CFLAGS) $(OMPFLAGS) $<

mmio.o: mmio.c
	$(CC) -o $@ -c $(CFLAGS) $<

# CPU targets
spmv: $(CPU_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

spmv-mpi: $(MPI_OBJS)
	$(MPICC) $(LDFLAGS) -o $@ $^

spmv-omp: $(OMP_OBJS)
	$(CC) $(LDFLAGS) $(OMPFLAGS) -o $@ $^

spmv-hybrid: $(HYBRID_OBJS)
	$(MPICC) $(LDFLAGS) $(OMPFLAGS) -o $@ $^

# CUDA targets
spmv-cuda: spmv-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-cuda.cu -o $@

spmv-mpi-cuda: spmv-mpi-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin $(MPICC) -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi -lstdc++

# Clean up
.PHONY: clean
clean:
	rm -f *.o spmv spmv-mpi spmv-omp spmv-hybrid spmv-cuda spmv-mpi-cuda