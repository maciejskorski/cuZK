#ifndef __BLS12_381_G1_CUH__
#define __BLS12_381_G1_CUH__

#include "bls12_381_init.cuh"
#include "../curve_utils.cuh"
#include "../../depends/libstl-cuda/vector.cuh"
#include "../../fields/field_utils.cuh"

namespace libff {

struct bls12_381_G1_params
{
    Fp_params<bls12_381_r_limbs>* fr_params;
    Fp_params<bls12_381_q_limbs>* fq_params;

    size_t* wnaf_window_table;
    size_t wnaf_window_table_size;
    
    size_t* fixed_base_exp_window_table_length;
    size_t fixed_base_exp_window_table_length_size;

    bls12_381_Fq* G1_zero_X;
    bls12_381_Fq* G1_zero_Y;
    bls12_381_Fq* G1_zero_Z;
    bls12_381_Fq* G1_one_X;
    bls12_381_Fq* G1_one_Y;
    bls12_381_Fq* G1_one_Z;
};

class bls12_381_G1 {
public:
    typedef bls12_381_Fq base_field;
    typedef bls12_381_Fr scalar_field;

    bls12_381_G1_params* params;

    bls12_381_Fq X, Y, Z;

    __device__ bls12_381_G1() : params(nullptr) {}
    __device__ bls12_381_G1(bls12_381_G1_params* params);
    __device__ bls12_381_G1(bls12_381_G1_params* params, const bls12_381_Fq& X, const bls12_381_Fq& Y, const bls12_381_Fq& Z) : params(params), X(X), Y(Y), Z(Z) {};
    __device__ bls12_381_G1(const bls12_381_G1& other) = default;

    __device__ bls12_381_G1& operator=(const bls12_381_G1& other) = default;

    __device__ void to_affine_coordinates();
    __device__ void to_special();
    __device__ bool is_special() const;

    __device__ bool is_zero() const;

    __device__ bool operator==(const bls12_381_G1 &other) const;
    __device__ bool operator!=(const bls12_381_G1 &other) const;

    __device__ bls12_381_G1 operator+(const bls12_381_G1 &other) const;

    __device__ bls12_381_G1 operator-() const;
    __device__ bls12_381_G1 operator-(const bls12_381_G1 &other) const;
    
    __device__ bls12_381_G1 operator*(const unsigned long lhs) const;

    __device__ bls12_381_G1 dbl() const;
    __device__ bls12_381_G1 add(const bls12_381_G1 &other) const;
    __device__ bls12_381_G1 mixed_add(const bls12_381_G1 &other) const;

    __device__ bool is_well_formed() const;

    __device__ bls12_381_G1 zero() const;
    __host__ bls12_381_G1* zero_host();
    __device__ bls12_381_G1 one() const;
    __host__ bls12_381_G1* one_host();

    __device__ bls12_381_G1 random_element() const;

    __device__ size_t size_in_bits();
    __device__ bigint<bls12_381_q_limbs> base_field_char();
    __device__ bigint<bls12_381_r_limbs> order();

    __device__ void set_params(bls12_381_G1_params* params);

    __device__ void batch_to_special(libstl::vector<bls12_381_G1> &vec);
    __device__ void p_batch_to_special(libstl::vector<bls12_381_G1> &vec, size_t gridSize, size_t blockSize);
};

template<mp_size_t_ m>
__device__ inline bls12_381_G1 operator*(const bigint<m>& lhs, const bls12_381_G1& rhs)
{
    return scalar_mul<bls12_381_G1, m>(rhs, lhs);
}

template<mp_size_t_ m>
__device__ inline bls12_381_G1 operator*(const Fp_model<m>& lhs, const bls12_381_G1& rhs)
{
    return scalar_mul<bls12_381_G1, m>(rhs, lhs.as_bigint());
}

}

#include "bls12_381_g1.cu"

#endif