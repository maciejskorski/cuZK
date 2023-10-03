// nvcc -arch=sm_35 -std=c++17 -lnvidia-ml  ./benchmark.cu ../../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu ../../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu  -o benchmark && ./benchmark 25

//#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
//#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cuh"
//#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cuh"
//#include "../../depends/libff-cuda/fields/bigint_host.cuh"
//#include "../../depends/libff-cuda/fields/fp_host.cuh"
#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cuh"
//#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "../../depends/libff-cuda/curves/bls12_381/bls12_381_pp.cuh"
//#include "../../depends/libff-cuda/fields/bigint_host.cuh"
using namespace libff;

//#include "../../depends/libstl-cuda/memory.cuh"
//#include "../../depends/libstl-cuda/vector.cuh"

#include <iostream>
#include <nvml.h>
using namespace std;


template<typename ppT>
struct MSM_params
{
    libstl::vector<libff::Fr<ppT>> vf;
    libstl::vector<libff::G1<ppT>> vg;
};

struct instance_params
{
    bls12_381_Fr instance;
    bls12_381_G1 g1_instance;
    bls12_381_G2 g2_instance;
    bls12_381_GT gt_instance;
};

struct h_instance_params
{
    bls12_381_Fr_host h_instance;
    bls12_381_G1_host h_g1_instance;
    bls12_381_G2_host h_g2_instance;
    bls12_381_GT_host h_gt_instance;
};



#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << endl;
        cerr << cudaGetErrorString(err) << endl;
    }
}

__global__ void generate_MP(libstl::vector<Fr<bls12_381_pp>>& vf, libstl::vector<G1<bls12_381_pp>>& vg, size_t size)
{
    for (int i=0; i<size; i++) {
        vf[i]  = libff::bls12_381_Fr(&bls12_381_fp_params_r);
        vg[i]  = libff::bls12_381_G1(&g1_params);
    };
}

int main(int argc, char* argv[])
{
    // initialize NVML objects for CUDA profiling
    nvmlReturn_t status;
    nvmlDevice_t device_handle;
    unsigned long long energy_start, energy_end;
    long long energy_total;
    cudaEvent_t time_start, time_stop;
    float time_total;
    unsigned int memClock;
    unsigned int ClockFreqNumber=200;
    unsigned int ClockFreqs[200];
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device_handle);
    nvmlDeviceGetApplicationsClock(device_handle, NVML_CLOCK_MEM, &memClock);
    nvmlDeviceGetSupportedGraphicsClocks(device_handle, memClock, &ClockFreqNumber, ClockFreqs );

    // prepare data
    bls12_381_pp_host::init_public_params();
    //libff::Fr<bls12_381_pp> f = bls12_381_Fr.random_element();
    //libff::G1<bls12_381_pp> g = bls12_381_G1.random_element();
    //MSM_params<bls12_381_pp> mp;
    MSM_params<bls12_381_pp> mp;
    printf("Size scalar+point=%d",sizeof(mp));
    printf("Size scalar=%d,point=%d",sizeof(libff::Fr<bls12_381_pp>),sizeof(libff::G1<bls12_381_pp>));

    libstl::vector<Fr<bls12_381_pp>> mp_vf;
    libstl::vector<G1<bls12_381_pp>> mp_vg;
    generate_MP<<<1,1>>>(mp_vf,mp_vg, 10);
    //cudaMalloc( (void**)&mp[i], sizeof(MSM_params<bls12_381_pp>))

    //MSM_params<bls12_381_pp>* mp;
    //cudaMalloc( (void**)&mp, sizeof(MSM_params<bls12_381_pp>));
    CHECK_LAST_CUDA_ERROR();
    printf("Dupa\n");
    nvmlShutdown();
}