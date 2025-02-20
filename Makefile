all: spmv spmv-mpi spmv-omp spmv-hybrid
CC=gcc
MPICC=mpicc

FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG=-O3 -lm

# Add OpenMP flag
OMPFLAG=-fopenmp

# Object files
OBJS=spmv.o mmio.o 
MPI_OBJS=spmv-mpi.o mmio.o
OMP_OBJS=spmv-omp.o mmio.o
HYBRID_OBJS=spmv-hybrid.o mmio.o

# Corrected implicit rule for .c -> .o
%.o: %.c
	${CC} -o $@ -c ${FLAG} $<

# Special rules for each version
spmv-mpi.o: spmv-mpi.c
	${MPICC} -o $@ -c ${FLAG} $<

spmv-omp.o: spmv-omp.c
	${CC} -o $@ -c ${FLAG} ${OMPFLAG} $<

spmv-hybrid.o: spmv-hybrid.c
	${MPICC} -o $@ -c ${FLAG} ${OMPFLAG} $<   

# Build rules
spmv: ${OBJS}
	${CC} ${LDFLAG} -o $@ $^

spmv-mpi: ${MPI_OBJS}
	${MPICC} ${LDFLAG} -o $@ $^

spmv-omp: ${OMP_OBJS}
	${CC} ${LDFLAG} ${OMPFLAG} -o $@ $^

spmv-hybrid: ${HYBRID_OBJS}
	${MPICC} ${LDFLAG} ${OMPFLAG} -o $@ $^

.PHONY: clean
clean: 
	rm -f *.o spmv spmv-mpi spmv-omp spmv-hybrid   # Simplified cleanup


