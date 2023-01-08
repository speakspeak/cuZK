#ifndef __GROTH_CUH__
#define __GROTH_CUH__

#include "../../../depends/libff-cuda/curves/public_params.cuh"
#include "../../../depends/libff-cuda/scalar_multiplication/multiexp.cuh"
#include "../../../depends/libstl-cuda/vector.cuh"
#include "../../../depends/libstl-cuda/memory.cuh"
#include "../../../depends/libmatrix-cuda/sparse-matrix/ell.cuh"
#include "../../relations/constraint_satisfaction_problems/r1cs/r1cs.cuh"
#include "groth_params.cuh"
#include <vector>

namespace cuzk{

template<typename ppT>
class groth_proving_key {
public:
    libff::G1<ppT> alpha_g1;
    libff::G1<ppT> beta_g1;
    libff::G2<ppT> beta_g2;
    libff::G1<ppT> delta_g1;
    libff::G2<ppT> delta_g2;

    libstl::vector<libff::G1<ppT>> A_g1_query;
    libstl::vector<libff::G1<ppT>> B_g1_query;
    libstl::vector<libff::G2<ppT>> B_g2_query;
    libstl::vector<libff::G1<ppT>> H_g1_query;
    libstl::vector<libff::G1<ppT>> L_g1_query;

    __host__ __device__ groth_proving_key() {};
    __device__ groth_proving_key(const groth_proving_key<ppT> &other) = default;
    __device__ groth_proving_key(const libff::G1<ppT>& alpha_g1,
                                  const libff::G1<ppT>& beta_g1,
                                  const libff::G2<ppT>& beta_g2,
                                  const libff::G1<ppT>& delta_g1,
                                  const libff::G2<ppT>& delta_g2,
                                  libstl::vector<libff::G1<ppT>> A_g1_query,
                                  libstl::vector<libff::G1<ppT>> B_g1_query,
                                  libstl::vector<libff::G2<ppT>> B_g2_query,
                                  libstl::vector<libff::G1<ppT>> H_g1_query,
                                  libstl::vector<libff::G1<ppT>> L_g1_query):
    alpha_g1(alpha_g1),
    beta_g1(beta_g1),
    beta_g2(beta_g2),
    delta_g1(delta_g1),
    delta_g2(delta_g2),
    A_g1_query(A_g1_query),
    B_g1_query(B_g1_query),
    B_g2_query(B_g2_query),
    H_g1_query(H_g1_query),
    L_g1_query(L_g1_query)
    {};

    __device__ groth_proving_key(libff::G1<ppT>&& alpha_g1,
                                libff::G1<ppT>&& beta_g1,
                                libff::G2<ppT>&& beta_g2,
                                libff::G1<ppT>&& delta_g1,
                                libff::G2<ppT>&& delta_g2,
                                libstl::vector<libff::G1<ppT>>&& A_g1_query,
                                libstl::vector<libff::G1<ppT>>&& B_g1_query,
                                libstl::vector<libff::G2<ppT>>&& B_g2_query,
                                libstl::vector<libff::G1<ppT>>&& H_g1_query,
                                libstl::vector<libff::G1<ppT>>&& L_g1_query):
    alpha_g1(libstl::move(alpha_g1)),
    beta_g1(libstl::move(beta_g1)),
    beta_g2(libstl::move(beta_g2)),
    delta_g1(libstl::move(delta_g1)),
    delta_g2(libstl::move(delta_g2)),
    A_g1_query(libstl::move(A_g1_query)),
    B_g1_query(libstl::move(B_g1_query)),
    B_g2_query(libstl::move(B_g2_query)),
    H_g1_query(libstl::move(H_g1_query)),
    L_g1_query(libstl::move(L_g1_query))
    {};

    __device__ groth_proving_key<ppT>& operator=(const groth_proving_key<ppT> &other) = default;

    __device__ groth_proving_key<ppT>& operator=(groth_proving_key<ppT> &&other)
    {
        alpha_g1 = libstl::move(other.alpha_g1);
        beta_g1 = libstl::move(other.beta_g1);
        beta_g2 = libstl::move(other.beta_g2);
        delta_g1 = libstl::move(other.delta_g1);
        delta_g2 = libstl::move(other.delta_g2);
        A_g1_query = libstl::move(other.A_g1_query);
        B_g1_query = libstl::move(other.B_g1_query);
        B_g2_query = libstl::move(other.B_g2_query);
        H_g1_query = libstl::move(other.H_g1_query);
        L_g1_query = libstl::move(other.L_g1_query);

        return *this;
    }
};

template<typename ppT>
struct groth_proving_key_host {

    libff::G1<ppT> alpha_g1;
    libff::G1<ppT> beta_g1;
    libff::G2<ppT> beta_g2;
    libff::G1<ppT> delta_g1;
    libff::G2<ppT> delta_g2;

    libstl::vector<libff::G1<ppT>> A_g1_query;
    libstl::vector<libff::G1<ppT>> B_g1_query;
    libstl::vector<libff::G2<ppT>> B_g2_query;
    libstl::vector<libff::G1<ppT>> H_g1_query;
    libstl::vector<libff::G1<ppT>> L_g1_query;
};

template<typename ppT_host, typename ppT_device>
void groth_proving_key_device2host(groth_proving_key_host<ppT_host>* hpk, groth_proving_key<ppT_device>* dpk);


template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device1(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device2(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device3(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device4(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device5(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);



template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance);


template<typename ppT>
class groth_verification_key {
public:
    libff::GT<ppT> alpha_g1_beta_g2;
    libff::G2<ppT> gamma_g2;
    libff::G2<ppT> delta_g2;

    libstl::vector<libff::G1<ppT>> gamma_ABC_g1;

    __host__ __device__ groth_verification_key() {};
    __device__ groth_verification_key(const groth_verification_key<ppT> &other) = default;
    __device__ groth_verification_key(const libff::GT<ppT> &alpha_g1_beta_g2,
                                       const libff::G2<ppT> &gamma_g2,
                                       const libff::G2<ppT> &delta_g2,
                                       libstl::vector<libff::G1<ppT>> &gamma_ABC_g1) :
        alpha_g1_beta_g2(alpha_g1_beta_g2),
        gamma_g2(gamma_g2),
        delta_g2(delta_g2),
        gamma_ABC_g1(gamma_ABC_g1)
        {};

    __device__  groth_verification_key(libff::GT<ppT> &&alpha_g1_beta_g2,
                                       libff::G2<ppT> &&gamma_g2,
                                       libff::G2<ppT> &&delta_g2,
                                       libstl::vector<libff::G1<ppT>> &&gamma_ABC_g1) :
        alpha_g1_beta_g2(libstl::move(alpha_g1_beta_g2)),
        gamma_g2(libstl::move(gamma_g2)),
        delta_g2(libstl::move(delta_g2)),
        gamma_ABC_g1(libstl::move(gamma_ABC_g1))
    {};

    __device__ groth_verification_key<ppT>& operator=(const groth_verification_key<ppT> &other) = default;

    __device__ groth_verification_key<ppT>& operator=(groth_verification_key<ppT> &&other)
    {
        alpha_g1_beta_g2 = libstl::move(other.alpha_g1_beta_g2);
        gamma_g2 = libstl::move(other.gamma_g2);
        delta_g2 = libstl::move(other.delta_g2);
        gamma_ABC_g1 = libstl::move(other.gamma_ABC_g1);

        return *this;
    }
};


template<typename ppT>
struct groth_verification_key_host {
    libff::GT<ppT> alpha_g1_beta_g2;
    libff::G2<ppT> gamma_g2;
    libff::G2<ppT> delta_g2;

    libstl::vector<libff::G1<ppT>> gamma_ABC_g1;
};

template<typename ppT_host, typename ppT_device>
void groth_verification_key_device2host(groth_verification_key_host<ppT_host>* hvk, groth_verification_key<ppT_device>* dvk);

template<typename ppT_host, typename ppT_device>
void groth_verification_key_host2device(groth_verification_key<ppT_device>* dvk, groth_verification_key_host<ppT_host>* hvk,
                                                libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, libff::GT<ppT_device>* gt_instance);


template<typename ppT>
class groth_processed_verification_key {
public:
    libff::GT<ppT> vk_alpha_g1_beta_g2;
    libff::G2_precomp<ppT> vk_gamma_g2_precomp;
    libff::G2_precomp<ppT> vk_delta_g2_precomp;
};


template<typename ppT>
class groth_keypair {
public:
    groth_proving_key<ppT> pk;
    groth_verification_key<ppT> vk;

    __host__ __device__ groth_keypair() {};

    __device__ groth_keypair(const groth_proving_key<ppT> &pk,
                              const groth_verification_key<ppT> &vk) :
        pk(pk),
        vk(vk)
    {};


    __device__ groth_keypair(groth_proving_key<ppT> &&pk,
                            groth_verification_key<ppT> &&vk) :
        pk(std::move(pk)),
        vk(std::move(vk))
    {};

    __device__ groth_keypair<ppT>& operator=(const groth_keypair<ppT> &other) = default;

    __device__ groth_keypair<ppT>& operator=(groth_keypair<ppT> &&other)
    {
        pk = std::move(other.pk);
        vk = std::move(other.vk);

        return *this;
    }
};

template<typename ppT>
struct groth_keypair_host {
    groth_proving_key_host<ppT> pk;
    groth_verification_key_host<ppT> vk;
};

template<typename ppT_host, typename ppT_device>
void groth_keypair_device2host(groth_keypair_host<ppT_host>* hkp, groth_keypair<ppT_device>* dkp);

template<typename ppT_host, typename ppT_device>
void groth_keypair_host2device(groth_keypair<ppT_device>* dkp, groth_keypair_host<ppT_host>* hkp, 
                                    libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, libff::GT<ppT_device>* gt_instance);



template<typename ppT>
class groth_proof {
public:
    libff::G1<ppT> g_A;
    libff::G2<ppT> g_B;
    libff::G1<ppT> g_C;

    __device__ groth_proof(){};


    __device__ groth_proof(const libff::G1<ppT> &g_A,
                            const libff::G2<ppT> &g_B,
                            const libff::G1<ppT> &g_C) :
        g_A(g_A),
        g_B(g_B),
        g_C(g_C)
    {};
};

template<typename ppT>
void groth_proof_device2host(groth_proof<ppT>* hpf, groth_proof<ppT>* dpf);

template<typename ppT>
void groth_proof_host2device(groth_proof<ppT>* dpf, groth_proof<ppT>* hpf, 
                                    libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance);



template<typename ppT>
void groth_generator(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip);


template<typename ppT>
void groth_prover(r1cs_constraint_system_host<libff::Fr<ppT>>* hcs, groth_proving_key_host<ppT>* hpk, r1cs_primary_input_host<libff::Fr<ppT>>* hpi, r1cs_auxiliary_input_host<libff::Fr<ppT>>* hai,
                            instance_params* ip);



}

#include "groth.cu"

#endif

