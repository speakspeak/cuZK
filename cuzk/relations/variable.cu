#ifndef __VARIABLE_CU__
#define __VARIABLE_CU__

namespace cuzk {


// template<typename FieldT>
// __device__ linear_term<FieldT> variable<FieldT>::operator*(const integer_coeff_t int_coeff, const FieldT& instance) const
// {
//     return linear_term<FieldT>(*this, int_coeff, instance);
// }

// template<typename FieldT>
// linear_term<FieldT> variable<FieldT>::operator*(const FieldT &field_coeff) const
// {
//     const FieldT& instance = field_coeff;
//     return linear_term<FieldT>(*this, field_coeff, instance);
// }

// template<typename FieldT>
// linear_combination<FieldT> variable<FieldT>::operator+(const linear_combination<FieldT> &other) const
// {
//     linear_combination<FieldT> result(other.instance);

//     result.add_term(*this);
//     result.terms.insert(result.terms.begin(), other.terms.begin(), other.terms.end());

//     return result;
// }

// template<typename FieldT>
// linear_combination<FieldT> variable<FieldT>::operator-(const linear_combination<FieldT> &other) const
// {
//     return (*this) + (-other);
// }

// template<typename FieldT>
// linear_term<FieldT> variable<FieldT>::operator-() const
// {
//     return linear_term<FieldT>(*this, -FieldT::one());
// }

// template<typename FieldT>
// bool variable<FieldT>::operator==(const variable<FieldT> &other) const
// {
//     return (this->index == other.index);
// }

// template<typename FieldT>
// linear_term<FieldT> operator*(const integer_coeff_t int_coeff, const variable<FieldT> &var)
// {
//     return linear_term<FieldT>(var, int_coeff);
// }

// template<typename FieldT>
// linear_term<FieldT> operator*(const FieldT &field_coeff, const variable<FieldT> &var)
// {
//     return linear_term<FieldT>(var, field_coeff);
// }

// template<typename FieldT>
// linear_combination<FieldT> operator+(const integer_coeff_t int_coeff, const variable<FieldT> &var)
// {
//     return linear_combination<FieldT>(int_coeff) + var;
// }

// template<typename FieldT>
// linear_combination<FieldT> operator+(const FieldT &field_coeff, const variable<FieldT> &var)
// {
//     return linear_combination<FieldT>(field_coeff) + var;
// }

// template<typename FieldT>
// linear_combination<FieldT> operator-(const integer_coeff_t int_coeff, const variable<FieldT> &var)
// {
//     return linear_combination<FieldT>(int_coeff) - var;
// }

// template<typename FieldT>
// linear_combination<FieldT> operator-(const FieldT &field_coeff, const variable<FieldT> &var)
// {
//     return linear_combination<FieldT>(field_coeff) - var;
// }

template<typename FieldT>
__device__ linear_term<FieldT>::linear_term(const FieldT& instance) :
    coeff(instance.params)
{
}

template<typename FieldT>
__device__ linear_term<FieldT>::linear_term(const variable<FieldT> &var, const FieldT& instance) :
    index(var.index), coeff(instance.one())
{
}

template<typename FieldT>
__device__ linear_term<FieldT>::linear_term(const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance) :
    index(var.index), coeff(FieldT(instance.params, int_coeff))
{
}

template<typename FieldT>
__device__ linear_term<FieldT>::linear_term(const variable<FieldT> &var, const FieldT &coeff, const FieldT& instance) :
    index(var.index), coeff(coeff)
{
}

template<typename FieldT>
__device__ linear_term<FieldT>& linear_term<FieldT>::operator=(const linear_term<FieldT>& other)
{
    if(this != &other)
    {
        index = other.index;
        coeff = other.coeff;
    }
    return *this;
}

template<typename FieldT>
__device__ linear_term<FieldT> linear_term<FieldT>::operator*(const integer_coeff_t int_coeff) const
{
    return (this->operator*(FieldT(int_coeff)));
}

template<typename FieldT>
__device__ linear_term<FieldT> linear_term<FieldT>::operator*(const FieldT &field_coeff) const
{
    const FieldT& instance = this->coeff;
    return linear_term<FieldT>(this->index, field_coeff * this->coeff, instance); 
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt)
{
    const FieldT& instance = lt.coeff;
    return linear_combination<FieldT>(int_coeff, instance) + lt;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const FieldT &field_coeff, const linear_term<FieldT> &lt)
{
    const FieldT& instance = lt.coeff;
    return linear_combination<FieldT>(field_coeff, instance) + lt;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt)
{
    const FieldT& instance = lt.coeff;
    return linear_combination<FieldT>(int_coeff, instance) - lt;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const FieldT &field_coeff, const linear_term<FieldT> &lt)
{
    const FieldT& instance = lt.coeff;
    return linear_combination<FieldT>(field_coeff, instance) - lt;
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_term<FieldT>::operator+(const linear_combination<FieldT> &other) const
{
    return linear_combination<FieldT>(*this) + other;
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_term<FieldT>::operator-(const linear_combination<FieldT> &other) const
{
    return (*this) + (-other);
}

template<typename FieldT>
__device__ linear_term<FieldT> linear_term<FieldT>::operator-() const
{
    const FieldT& instance = this->coeff;
    return linear_term<FieldT>(this->index, -this->coeff, instance);
}

template<typename FieldT>
__device__ bool linear_term<FieldT>::operator==(const linear_term<FieldT> &other) const
{
    return (this->index == other.index &&
            this->coeff == other.coeff);
}

template<typename FieldT>
__device__ linear_term<FieldT> operator*(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt)
{
    return FieldT(int_coeff) * lt;
}

template<typename FieldT>
__device__ linear_term<FieldT> operator*(const FieldT &field_coeff, const linear_term<FieldT> &lt)
{
    const FieldT& instance = field_coeff.coeff;
    return linear_term<FieldT>(lt.index, field_coeff * lt.coeff, instance);
}

template<typename FieldT>
__device__ linear_combination<FieldT>::linear_combination(const integer_coeff_t int_coeff, const FieldT& instance)
{
    this->add_term(linear_term<FieldT>(0, int_coeff), instance);
}

template<typename FieldT>
__device__ linear_combination<FieldT>::linear_combination(const FieldT &field_coeff, const FieldT& instance)
{
    this->add_term(linear_term<FieldT>(0, field_coeff), instance);
}

template<typename FieldT>
__device__ linear_combination<FieldT>::linear_combination(const variable<FieldT> &var, const FieldT& instance)
{
    this->add_term(var, instance);
}

template<typename FieldT>
__device__ linear_combination<FieldT>::linear_combination(const linear_term<FieldT> &lt)
{
    this->add_term(lt);
}

template<typename FieldT>
__device__ linear_combination<FieldT>::linear_combination(const linear_combination<FieldT>& other)
{
    this->vterms = other.vterms;
}

template<typename FieldT>
__device__ linear_combination<FieldT>& linear_combination<FieldT>::operator=(const linear_combination<FieldT>& other)
{
    if(this != &other)
    {
        this->vterms = other.vterms;
    }
    return *this;
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::reserve_term(const size_t size, const FieldT& instance)
{
    vterms.resize(size, linear_term<FieldT>(instance));
}


template<typename FieldT>
__device__ void linear_combination<FieldT>::add_term(const variable<FieldT> &var, const FieldT& instance)
{
    vterms.resize(vterms.size()+1, linear_term<FieldT>(var, 1 , instance));
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::add_term(const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance)
{
    vterms.resize(vterms.size()+1, linear_term<FieldT>(var, int_coeff, instance));
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::add_term(const variable<FieldT> &var, const FieldT &field_coeff, const FieldT& instance)
{
    vterms.resize(vterms.size()+1, linear_term<FieldT>(var, field_coeff, instance));

}

template<typename FieldT>
__device__ void linear_combination<FieldT>::add_term(const linear_term<FieldT> &lt)
{
    vterms.resize(vterms.size()+1, lt);
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::set_term(const size_t idx, const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance)
{
    vterms[idx] = linear_term<FieldT>(var, int_coeff, instance);
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::set_term(const size_t idx, const variable<FieldT> &var, const FieldT& instance)
{
    vterms[idx] = linear_term<FieldT>(var, 1 , instance);
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::set_term(const size_t idx, const variable<FieldT> &var, const FieldT &field_coeff, const FieldT& instance)
{
    vterms[idx] = linear_term<FieldT>(var, field_coeff, instance);
}

template<typename FieldT>
__device__ void linear_combination<FieldT>::set_term(const size_t idx, const linear_term<FieldT> &lt)
{
    vterms[idx] = lt;
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_combination<FieldT>::operator*(const integer_coeff_t int_coeff) const
{
    if(this->vterms.size() == 0) return linear_combination<FieldT>();
    else
    {
        const FieldT& instance = this->vterms[0].coeff;
        return (*this) * FieldT(instance.params, int_coeff);
    }
}


template<typename FieldT>
__device__ FieldT linear_combination<FieldT>::evaluate(const libstl::vector<FieldT>& assignment, const FieldT& instance) const
{
    FieldT acc = instance.zero();
    for(int i=0; i < vterms.size(); i++)
    {
        acc += (vterms[i].index == 0 ? instance.one() : assignment[vterms[i].index-1]) * vterms[i].coeff;
    }
    return acc;
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_combination<FieldT>::operator*(const FieldT &field_coeff) const
{
    linear_combination<FieldT> result;
    for (int i=0; i < vterms.size(); i++)
    {
        result.add_term(vterms[i] * field_coeff);
    }
    return result;
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_combination<FieldT>::operator+(const linear_combination<FieldT> &other) const
{
    

    linear_combination<FieldT> result;

    int i1 = 0;
    int i2 = 0;

    /* invariant: it1 and it2 always point to unprocessed items in the corresponding linear combinations */
    while(i1 != this->vterms.size() && i2 != other.vterms.size())
    {
        if (this->vterms[i1].index < other.vterms[i2].index)
        {
            result.add_term(vterms[i1]);
            ++i1;
        }
        else if (this->vterms[i1].index > other.vterms[i2].index)
        {
            result.add_term(other.vterms[i2]);
            ++i2;
        }
        else
        {
            /* it1->index == it2->index */
            result.add_term(linear_term<FieldT>(variable<FieldT>(this->vterms[i1].index), this->vterms[i1].coeff + other.vterms[i2].coeff));
            ++i1;
            ++i2;
        }
    }

    if (i1 != this->vterms.size())
    {
        for(; i1 < this->vterms.size(); i1++)
        {
            result.add_term(this->vterms[i1]);
        }
    }
    else
    {
        for(; i2 < other.vterms.size(); i2++)
        {
            result.add_term(other.vterms[i2]);
        }
    }

    return result;
}


template<typename FieldT>
__device__ linear_combination<FieldT> linear_combination<FieldT>::operator-(const linear_combination<FieldT> &other) const
{
    return (*this) + (-other);
}

template<typename FieldT>
__device__ linear_combination<FieldT> linear_combination<FieldT>::operator-() const
{
    if(this->vterms.size() == 0) return linear_combination<FieldT>();
    else
    {
        const FieldT& instance = this->vterms[0].coeff;
        return (*this) * (-instance.one());
    }
}

template<typename FieldT>
__device__ bool linear_combination<FieldT>::operator==(const linear_combination<FieldT> &other) const
{
    if(this->vterms.size() != other.vterms.size()) return false;
    else
    {
        for(int i=0; i < this->vterms.size(); i++)
        {
            if(this->vterms[i] != other.vterms[i]) return false;
        }
    }
    return true;
}

template<typename FieldT>
__device__ bool linear_combination<FieldT>::is_valid(const size_t num_variables) const
{
    /* check that all terms in linear combination are sorted */
    for (size_t i = 1; i < vterms.size(); ++i)
    {
        if (vterms[i-1].index >= vterms[i].index)
        {
            return false;
        }
    }

    /* check that the variables are in proper range. as the variables
       are sorted, it suffices to check the last term */
    if (vterms[vterms.size()-1].index >= num_variables)
    {
        return false;
    }

    return true;
}


template<typename FieldT>
__device__ linear_combination<FieldT> operator*(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc)
{
    return lc * int_coeff;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator*(const FieldT &field_coeff, const linear_combination<FieldT> &lc)
{
    return lc * field_coeff;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc)
{
    return linear_combination<FieldT>(int_coeff) + lc;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const FieldT &field_coeff, const linear_combination<FieldT> &lc)
{
    return linear_combination<FieldT>(field_coeff) + lc;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc)
{
    return linear_combination<FieldT>(int_coeff) - lc;
}

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const FieldT &field_coeff, const linear_combination<FieldT> &lc)
{
    return linear_combination<FieldT>(field_coeff) - lc;
}


}

#endif