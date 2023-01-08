#ifndef __PUBLIC_PARAMS_CUH__
#define __PUBLIC_PARAMS_CUH__

namespace libff{

template<typename EC_ppT>
using Fr = typename EC_ppT::Fp_type;
template<typename EC_ppT>
using G1 = typename EC_ppT::G1_type;
template<typename EC_ppT>
using G2 = typename EC_ppT::G2_type;
template<typename EC_ppT>
using G1_precomp = typename EC_ppT::G1_precomp_type;
template<typename EC_ppT>
using G2_precomp = typename EC_ppT::G2_precomp_type;
template<typename EC_ppT>
using Fq = typename EC_ppT::Fq_type;
template<typename EC_ppT>
using Fqe = typename EC_ppT::Fqe_type;
template<typename EC_ppT>
using Fqk = typename EC_ppT::Fqk_type;
template<typename EC_ppT>
using GT = typename EC_ppT::GT_type;

}

#endif