# Makefile for GPU GroupBy Project
# EE-5351 Fall 2018
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11
ifdef dbg
	NVCC_FLAGS  += -g -G
	CXX_FLAGS += -DDEBUG
else
	NVCC_FLAGS  += -O3
endif

ifdef NOPRINT
	CXX_FLAGS += -DNOPRINT
endif

ifdef PRIV
	CXX_FLAGS += -DPRIVATIZATION
endif

ifdef TESLA
	CXX_FLAGS += -DTESLA
endif

ifdef GPU_SAMPLE
	CXX_FLAGS += -DGPU_SAMPLE
endif

ifdef CPU_SAMPLE
	CXX_FLAGS += -DCPU_SAMPLE
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE_DECIMAL = decimal
OBJ_DECIMAL = main.o sum_reduce.o decimal.o

default: $(EXE_DECIMAL)


decimal.o: decimal.cpp decimal.h
	$(NVCC) -c -o $@ decimal.cpp $(NVCC_FLAGS) $(CXX_FLAGS)

main.o: main.cpp sum_reduce.cuh decimal.h
	$(NVCC) -c -o $@ main.cpp $(NVCC_FLAGS) $(CXX_FLAGS)

sum_reduce.o: sum_reduce.cu sum_reduce.cuh decimal.h
	$(NVCC) -c -o $@ sum_reduce.cu $(NVCC_FLAGS) $(CXX_FLAGS)


$(EXE_DECIMAL): $(OBJ_DECIMAL)
	$(NVCC) $(OBJ_DECIMAL) -o $(EXE_DECIMAL) $(LD_FLAGS) $(NVCC_FLAGS) $(CXX_FLAG)

clean:
	rm -rf *.o $(EXE_DECIMAL)
