all: msm

precompile_host:
	nvcc -arch=sm_35 -std=c++17 --threads 0 -c ../../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cu -o bls12_381_g1_host.o
	nvcc -arch=sm_35 -std=c++17 --threads 0 -c ../../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cu -o bls12_381_g2_host.o
	nvcc -arch=sm_35 -std=c++17 --threads 0 -c ../../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu -o bls12_381_init_host.o
	nvcc -arch=sm_35 -std=c++17 --threads 0 -c ../../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu -o bls12_381_pp_host.o

precompile_device:
	nvcc -arch=sm_35 -std=c++17 -rdc=true --threads 0 --extended-lambda -c ../../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o mini-mp-cuda.o
	nvcc -arch=sm_35 -std=c++17 -rdc=true --threads 0 --extended-lambda -c ../../depends/libstl-cuda/memory.cu -o memory.o

msm:
	nvcc -arch=sm_35 -std=c++17 -rdc=true --threads 0 --extended-lambda -lnvidia-ml -lgmp ./benchmark.cu ./bls12_381_init_host.o ./bls12_381_pp_host.o  ./bls12_381_g1_host.o ./bls12_381_g2_host.o ./mini-mp-cuda.o ./memory.o -o benchmark
