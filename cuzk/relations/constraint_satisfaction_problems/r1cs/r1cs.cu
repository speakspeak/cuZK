#ifndef __R1CS_CU__
#define __R1CS_CU__

#include <assert.h>


namespace cuzk {


template<typename FieldT>
__device__ r1cs_constraint<FieldT>::r1cs_constraint(const FieldT& instance):a(instance), b(instance), c(instance)
{
}

template<typename FieldT>
__device__ r1cs_constraint<FieldT>::r1cs_constraint(const linear_combination<FieldT> &a,
                                         const linear_combination<FieldT> &b,
                                         const linear_combination<FieldT> &c) :
    a(a), b(b), c(c)
{
}

template<typename FieldT>
__device__ r1cs_constraint<FieldT>::r1cs_constraint(const r1cs_constraint& other):a(other.a), b(other.b), c(other.c)
{
}

template<typename FieldT>
__device__ r1cs_constraint<FieldT>& r1cs_constraint<FieldT>::operator=(const r1cs_constraint<FieldT>& other)
{
    
    if(this != &other)
    {
        a = other.a;
        b = other.b;
        c = other.c;
    }
    return *this;
}


template<typename FieldT>
__device__ bool r1cs_constraint<FieldT>::operator==(const r1cs_constraint<FieldT> &other) const
{
    return (this->a == other.a &&
            this->b == other.b &&
            this->c == other.c);
}

template<typename FieldT>
__device__ r1cs_constraint_system<FieldT>::r1cs_constraint_system(libstl::vector<r1cs_constraint<FieldT>>&& vconstraints, size_t primary_input_size, size_t auxiliary_input_size)
:vconstraints(libstl::move(vconstraints)), primary_input_size(primary_input_size), auxiliary_input_size(auxiliary_input_size)
{
    ma.row_size = this->vconstraints.size();
    mb.row_size = this->vconstraints.size();
    mc.row_size = this->vconstraints.size();

    ma.row_ptr.resize(ma.row_size + 1, 0);
    mb.row_ptr.resize(mb.row_size + 1, 0);
    mc.row_ptr.resize(mc.row_size + 1, 0);

    size_t a_elements_num = 0;
    size_t b_elements_num = 0;
    size_t c_elements_num = 0;
    for(size_t i=0; i < this->vconstraints.size(); i++)
    {
        a_elements_num += this->vconstraints[i].a.vterms.size();
        ma.row_ptr[i + 1] = a_elements_num;
        b_elements_num += this->vconstraints[i].b.vterms.size();
        mb.row_ptr[i + 1] = b_elements_num;
        c_elements_num += this->vconstraints[i].c.vterms.size();
        mc.row_ptr[i + 1] = c_elements_num;
    }

    ma.col_idx.resize(a_elements_num);
    mb.col_idx.resize(b_elements_num);
    mc.col_idx.resize(c_elements_num);
    ma.data.resize(a_elements_num);
    mb.data.resize(b_elements_num);
    mc.data.resize(c_elements_num);

    ma.col_size = 0;
    mb.col_size = 0;
    mc.col_size = 0;


    for(size_t i = 0; i < ma.row_size; i++)
    {
        for(size_t j=0; j < ma.row_ptr[i+1] - ma.row_ptr[i]; j++)
        {
            ma.col_idx[j + ma.row_ptr[i]] = this->vconstraints[i].a.vterms[j].index;
            ma.data[j + ma.row_ptr[i]] = this->vconstraints[i].a.vterms[j].coeff;
            if(ma.col_size < this->vconstraints[i].a.vterms[j].index + 1) ma.col_size = this->vconstraints[i].a.vterms[j].index + 1;
        }
    }

    for(size_t i = 0; i < mb.row_size; i++)
    {
        for(size_t j=0; j < mb.row_ptr[i+1] - mb.row_ptr[i]; j++)
        {
            mb.col_idx[j + mb.row_ptr[i]] = this->vconstraints[i].b.vterms[j].index;
            mb.data[j + mb.row_ptr[i]] = this->vconstraints[i].b.vterms[j].coeff;
            if(mb.col_size < this->vconstraints[i].b.vterms[j].index + 1) mb.col_size = this->vconstraints[i].b.vterms[j].index + 1;
        }
    }

    for(size_t i = 0; i < mc.row_size; i++)
    {
        for(size_t j=0; j < mc.row_ptr[i+1] - mc.row_ptr[i]; j++)
        {
            mc.col_idx[j + mc.row_ptr[i]] = this->vconstraints[i].c.vterms[j].index;
            mc.data[j + mc.row_ptr[i]] = this->vconstraints[i].c.vterms[j].coeff;
            if(mc.col_size < this->vconstraints[i].c.vterms[j].index + 1) mc.col_size = this->vconstraints[i].c.vterms[j].index + 1;
        }
    }

    size_t max_col_size = 0;
    if(ma.col_size > max_col_size) max_col_size = ma.col_size;
    if(mb.col_size > max_col_size) max_col_size = mb.col_size;
    if(mc.col_size > max_col_size) max_col_size = mc.col_size;

    ma.col_size = max_col_size;
    mb.col_size = max_col_size;
    mc.col_size = max_col_size;
}

template<typename FieldT>
__device__ r1cs_constraint_system<FieldT>::r1cs_constraint_system(const r1cs_constraint_system& other)
{
    this->primary_input_size = other.primary_input_size;
    this->auxiliary_input_size = other.auxiliary_input_size;
    this->vconstraints = other.vconstraints;
    this->ma = other.ma;
    this->mb = other.mb;
    this->mc = other.mc;
}

template<typename FieldT>
__device__ r1cs_constraint_system<FieldT>::r1cs_constraint_system(r1cs_constraint_system<FieldT>&& other):
vconstraints(libstl::move(other.vconstraints)), ma(libstl::move(other.ma)), mb(libstl::move(other.mb)), mc(libstl::move(other.mc)), primary_input_size(other.primary_input_size), auxiliary_input_size(other.auxiliary_input_size)
{
}

template<typename FieldT>
__device__ r1cs_constraint_system<FieldT>& r1cs_constraint_system<FieldT>::operator=(const r1cs_constraint_system<FieldT> &other)
{
    this->primary_input_size = other.primary_input_size;
    this->auxiliary_input_size = other.auxiliary_input_size;
    this->vconstraints = other.vconstraints;
    this->ma = other.ma;
    this->mb = other.mb;
    this->mc = other.mc;

    return *this;
}

template<typename FieldT>
__device__ r1cs_constraint_system<FieldT>& r1cs_constraint_system<FieldT>::operator=(r1cs_constraint_system<FieldT> &&other)
{
    this->primary_input_size = other.primary_input_size;
    this->auxiliary_input_size = other.auxiliary_input_size;
    this->vconstraints = libstl::move(other.vconstraints);
    this->ma = libstl::move(other.ma);
    this->mb = libstl::move(other.mb);
    this->mc = libstl::move(other.mc);

    return *this;
}

template<typename FieldT>
__device__ size_t r1cs_constraint_system<FieldT>::num_inputs() const
{
    return primary_input_size;
}

template<typename FieldT>
__device__ size_t r1cs_constraint_system<FieldT>::num_variables() const
{
    return primary_input_size + auxiliary_input_size;
}

template<typename FieldT>
__device__ size_t r1cs_constraint_system<FieldT>::num_constraints() const
{
    return ma.row_ptr.size() - 1;
    // return vconstraints.size();
}


template<typename FieldT>
__host__ size_t r1cs_constraint_system<FieldT>::num_inputs_host()
{
    size_t primary_input_size_host;
    libstl::get_host(&primary_input_size_host, &this->primary_input_size);

    return primary_input_size_host;
}


template<typename FieldT>
__host__ size_t r1cs_constraint_system<FieldT>::num_variables_host()
{
    size_t primary_input_size_host;
    size_t auxiliary_input_size_host;
    libstl::get_host(&primary_input_size_host, &this->primary_input_size);
    libstl::get_host(&auxiliary_input_size_host, &this->auxiliary_input_size_host);

    return primary_input_size_host + auxiliary_input_size_host;
}

template<typename FieldT>
__host__ size_t r1cs_constraint_system<FieldT>::num_constraints_host()
{
    return ma.row_ptr.size_host() - 1;
}

template<typename FieldT>
__device__ bool r1cs_constraint_system<FieldT>::is_valid() const
{
    if (this->num_inputs() > this->num_variables()) return false;

    for (size_t c = 0; c < this->num_constraints(); ++c)
    {
        if (!(vconstraints[c].a.is_valid(this->num_variables()) &&
              vconstraints[c].b.is_valid(this->num_variables()) &&
              vconstraints[c].c.is_valid(this->num_variables())))
        {
            return false;
        }
    }

    return true;
}



template<typename FieldT>
__device__ bool r1cs_constraint_system<FieldT>::is_satisfied(const libstl::vector<FieldT>& primary_input, const libstl::vector<FieldT>& auxiliary_input, const FieldT& instance) const
{
    assert(primary_input.size() == num_inputs());
    assert(primary_input.size() + auxiliary_input.size() == num_variables());

    libstl::vector<FieldT> full_variable_assignment(num_variables(), instance);
    for(int i=0; i<num_variables(); i++)
    {
        if(i < primary_input_size)
            full_variable_assignment[i] = primary_input[i];
        else
            full_variable_assignment[i] = auxiliary_input[i-primary_input.size()];
    }

    for (size_t c = 0; c < vconstraints.size(); ++c)
    {
        const FieldT ares = vconstraints[c].a.evaluate(full_variable_assignment, instance);
        const FieldT bres = vconstraints[c].b.evaluate(full_variable_assignment, instance);
        const FieldT cres = vconstraints[c].c.evaluate(full_variable_assignment, instance);

        if (!(ares*bres == cres))
        {
            return false;
        }
    }

    return true;
}

template<typename FieldT>
__device__ void r1cs_constraint_system<FieldT>::add_constraint(const r1cs_constraint<FieldT> &c, const FieldT& instance)
{
    vconstraints.resize(vconstraints.size()+1, c);

}

template<typename FieldT>
__device__ void r1cs_constraint_system<FieldT>::add_constraint(libstl::vector<r1cs_constraint<FieldT>> cs, const FieldT& instance)
{
    vconstraints.resize(vconstraints.size()+cs.size(), cs[0]);
    for(int i=0; i < cs.size(); i++)
    {
        vconstraints[vconstraints.size() + i] = cs[i];
    }
}


template<typename FieldT>
__device__ void r1cs_constraint_system<FieldT>::swap_AB_if_beneficial()
{
}

template<typename FieldT>
__device__ bool r1cs_constraint_system<FieldT>::operator==(const r1cs_constraint_system<FieldT> &other) const
{
    return (this->constraints == other.constraints &&
            this->primary_input_size == other.primary_input_size &&
            this->auxiliary_input_size == other.auxiliary_input_size);
}


template<typename FieldT_host, typename FieldT_device>
void r1cs_constraint_system_device2host(r1cs_constraint_system_host<FieldT_host>* hcs, r1cs_constraint_system<FieldT_device>* dcs)
{
    cudaMemcpy(&hcs->primary_input_size, &dcs->primary_input_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hcs->auxiliary_input_size, &dcs->auxiliary_input_size, sizeof(size_t), cudaMemcpyDeviceToHost);

    CSR_matrix_device2host(&hcs->ma, &dcs->ma);
    CSR_matrix_device2host(&hcs->mb, &dcs->mb);
    CSR_matrix_device2host(&hcs->mc, &dcs->mc);
}

template<typename FieldT_host, typename FieldT_device>
void r1cs_constraint_system_host2device(r1cs_constraint_system<FieldT_device>* dcs, r1cs_constraint_system_host<FieldT_host>* hcs, FieldT_device* instance)
{
    cudaMemcpy(&dcs->primary_input_size, &hcs->primary_input_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&dcs->auxiliary_input_size, &hcs->auxiliary_input_size, sizeof(size_t), cudaMemcpyHostToDevice);

    CSR_matrix_host2device(&dcs->ma, &hcs->ma);
    CSR_matrix_host2device(&dcs->mb, &hcs->mb);
    CSR_matrix_host2device(&dcs->mc, &hcs->mc);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dcs->ma.data.size())
            {
                dcs->ma.data[idx].set_params(instance->params);
                idx += gridDim.x * blockDim.x;
            }
            idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dcs->mb.data.size())
            {
                dcs->mb.data[idx].set_params(instance->params);
                idx += gridDim.x * blockDim.x;
            }
            idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dcs->mc.data.size())
            {
                dcs->mc.data[idx].set_params(instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}

template<typename FieldT_host, typename FieldT_device>
void r1cs_primary_input_device2host(r1cs_primary_input_host<FieldT_host>* hpi, r1cs_primary_input<FieldT_device>* dpi)
{
    vector_device2host(hpi, dpi);
}

template<typename FieldT_host, typename FieldT_device>
void r1cs_primary_input_host2device(r1cs_primary_input<FieldT_device>* dpi, r1cs_primary_input_host<FieldT_host>* hpi, FieldT_device* instance)
{
    vector_host2device(dpi, hpi);

    size_t gridSize = 512;
    size_t blockSize = 32;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpi->size())
            {
                (*dpi)[idx].set_params(instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}

template<typename FieldT_host, typename FieldT_device>
void r1cs_auxiliary_input_device2host(r1cs_auxiliary_input_host<FieldT_host>* hai, r1cs_auxiliary_input<FieldT_device>* dai)
{
    vector_device2host(hai, dai);
}

template<typename FieldT_host, typename FieldT_device>
void r1cs_auxiliary_input_host2device(r1cs_auxiliary_input<FieldT_device>* dai, r1cs_auxiliary_input_host<FieldT_host>* hai, FieldT_device* instance)
{
    vector_host2device(dai, hai);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dai->size())
            {
                (*dai)[idx].set_params(instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}


}




#endif
