#ifndef __CSR_CU__
#define __CSR_CU__

namespace libmatrix{


template<typename T, typename H>
void CSR_matrix_device2host(CSR_matrix_host<H>* hcsr, CSR_matrix<T>* dcsr)
{
    cudaMemcpy(&hcsr->row_size, &dcsr->row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hcsr->col_size, &dcsr->col_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    vector_device2host(&hcsr->row_ptr, &dcsr->row_ptr);
    vector_device2host(&hcsr->col_idx, &dcsr->col_idx);
    vector_device2host(&hcsr->data, &dcsr->data);
}



template<typename T, typename H>
void CSR_matrix_host2device(CSR_matrix<T>* dcsr, CSR_matrix_host<H>* hcsr)
{
    cudaMemcpy(&dcsr->row_size, &hcsr->row_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&dcsr->col_size, &hcsr->col_size, sizeof(size_t), cudaMemcpyHostToDevice);
    vector_host2device(&dcsr->row_ptr, &hcsr->row_ptr);
    vector_host2device(&dcsr->col_idx, &hcsr->col_idx);
    vector_host2device(&dcsr->data, &hcsr->data);
}


template<typename T>
void CSR_matrix_device2host(CSR_matrix_host<T>* hcsr, CSR_matrix<T>* dcsr)
{
    CSR_matrix_device2host<T, T>(hcsr, dcsr);
}

template<typename T>
void CSR_matrix_host2device(CSR_matrix<T>* dcsr, CSR_matrix_host<T>* hcsr)
{
    CSR_matrix_host2device<T, T>(dcsr, hcsr);
}

}


#endif