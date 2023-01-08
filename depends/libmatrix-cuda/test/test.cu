#include "../sparse-matrix/csr.cuh"
#include "../sparse-matrix/ell.cuh"
#include "../spmv/csr-scalar.cuh"
#include "../spmv/csr-vector.cuh"
#include "../spmv/csr-balanced.cuh"
#include "../transpose/transpose_csr2csr.cuh"
#include "../transpose/transpose_ell2csr.cuh"
#include "../depends/libstl-cuda/memory.cuh"

using namespace libmatrix;

//   Sparse Matrix 
//   row size = 5
//   col size = 6
//   non zero entries = 15

//   | 1 | 0 | 3 | 0 | 5 | 0 |
//   | 6 | 7 | 0 | 9 | 1 | 0 |
//   | 3 | 5 | 4 | 0 | 0 | 1 |
//   | 2 | 0 | 0 | 2 | 9 | 0 |
//   | 0 | 0 | 0 | 0 | 5 | 0 |

__device__ ELL_matrix<long>* ell_matrix_generate()
{
    ELL_matrix<long>* pmtx = libstl::create<ELL_matrix<long>>(ELL_matrix<long>());
    ELL_matrix<long>& mtx = *pmtx;

    mtx.max_row_length = 4;
    mtx.row_size = 5;
    mtx.col_size = 6;
    mtx.row_length.resize(mtx.row_size);
    mtx.col_idx.resize(mtx.max_row_length * mtx.row_size);
    mtx.data.resize(mtx.max_row_length * mtx.row_size);

    mtx.row_length[0] = 3;
    mtx.row_length[1] = 4;
    mtx.row_length[2] = 4;
    mtx.row_length[3] = 3;
    mtx.row_length[4] = 1;

    mtx.col_idx[0] = 0;
    mtx.col_idx[1] = 2;
    mtx.col_idx[2] = 4;
    mtx.col_idx[3] = mtx.col_size;
    mtx.col_idx[4] = 0;
    mtx.col_idx[5] = 1;
    mtx.col_idx[6] = 3;
    mtx.col_idx[7] = 4;
    mtx.col_idx[8] = 0;
    mtx.col_idx[9] = 1;
    mtx.col_idx[10] = 2;
    mtx.col_idx[11] = 5;
    mtx.col_idx[12] = 0;
    mtx.col_idx[13] = 3;
    mtx.col_idx[14] = 4;
    mtx.col_idx[15] = mtx.col_size;
    mtx.col_idx[16] = 4;
    mtx.col_idx[17] = mtx.col_size;
    mtx.col_idx[18] = mtx.col_size;
    mtx.col_idx[19] = mtx.col_size;

    mtx.data[0] = 1;
    mtx.data[1] = 3;
    mtx.data[2] = 5;
    mtx.data[3] = 0;
    mtx.data[4] = 6;
    mtx.data[5] = 7;
    mtx.data[6] = 9;
    mtx.data[7] = 1;
    mtx.data[8] = 3;
    mtx.data[9] = 5;
    mtx.data[10] = 4;
    mtx.data[11] = 1;
    mtx.data[12] = 2;
    mtx.data[13] = 2;
    mtx.data[14] = 9;
    mtx.data[15] = 0;
    mtx.data[16] = 5;
    mtx.data[17] = 0;
    mtx.data[18] = 0;
    mtx.data[19] = 0;

    return pmtx;
}


__device__ CSR_matrix<long>* csr_matrix_generate()
{
    CSR_matrix<long>* pmtx = libstl::create<CSR_matrix<long>>(CSR_matrix<long>());
    CSR_matrix<long>& mtx = *pmtx;

    mtx.row_size = 5;
    mtx.col_size = 6;
    mtx.row_ptr.resize(mtx.row_size + 1);
    mtx.col_idx.resize(15);
    mtx.data.resize(15);

    mtx.row_ptr[0] = 0;
    mtx.row_ptr[1] = 3;
    mtx.row_ptr[2] = 7;
    mtx.row_ptr[3] = 11;
    mtx.row_ptr[4] = 14;
    mtx.row_ptr[5] = 15;

    mtx.col_idx[0] = 0;
    mtx.col_idx[1] = 2;
    mtx.col_idx[2] = 4;
    mtx.col_idx[3] = 0;
    mtx.col_idx[4] = 1;
    mtx.col_idx[5] = 3;
    mtx.col_idx[6] = 4;
    mtx.col_idx[7] = 0;
    mtx.col_idx[8] = 1;
    mtx.col_idx[9] = 2;
    mtx.col_idx[10] = 5;
    mtx.col_idx[11] = 0;
    mtx.col_idx[12] = 3;
    mtx.col_idx[13] = 4;
    mtx.col_idx[14] = 4;

    mtx.data[0] = 1;
    mtx.data[1] = 3;
    mtx.data[2] = 5;
    mtx.data[3] = 6;
    mtx.data[4] = 7;
    mtx.data[5] = 9;
    mtx.data[6] = 1;
    mtx.data[7] = 3;
    mtx.data[8] = 5;
    mtx.data[9] = 4;
    mtx.data[10] = 1;
    mtx.data[11] = 2;
    mtx.data[12] = 2;
    mtx.data[13] = 9;
    mtx.data[14] = 5;

    return pmtx;
}


__global__ void warmup()
{

    size_t init_size = 100000;
    libstl::initAllocator(init_size);
    
    CSR_matrix<long>* pmtx = csr_matrix_generate();
    CSR_matrix<long>& mtx = *pmtx;

    ELL_matrix<long>* pell_mtx = ell_matrix_generate();
    ELL_matrix<long>& ell_mtx = *pell_mtx;

    libstl::vector<long>* pv = libstl::create<libstl::vector<long>>(libstl::vector<long>(6));
    libstl::vector<long>& v = *pv;
    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    v[3] = 1;
    v[4] = 1;
    v[5] = 1;
    long* zero = libstl::create<long>(0);

    libstl::vector<long>* pres = p_spmv_csr_balanced_vector_one<long>(mtx, *zero, 2, 2);
    libstl::vector<long>& res = *pres;

    // CSR_matrix<long>* p_transpose_mtx = p_transpose_csr2csr(mtx, 1, 32);
    // CSR_matrix<long>* p_transpose_mtx_from_ell = p_transpose_ell2csr(ell_mtx, 1, 1);

    for(size_t i=0; i<res.size(); i++) printf("res: %d\n", res[i]);
    // for(size_t i=0; i<p_transpose_mtx_from_ell->col_idx.size(); i++) printf("col idx: %d\n", p_transpose_mtx_from_ell->col_idx[i]);
    // for(size_t i=0; i<p_transpose_mtx_from_ell->data.size(); i++) printf("data: %d\n", p_transpose_mtx_from_ell->data[i]);
}

int main()
{
    warmup<<<1, 1>>>();
    cudaDeviceReset();
    return 0;
}