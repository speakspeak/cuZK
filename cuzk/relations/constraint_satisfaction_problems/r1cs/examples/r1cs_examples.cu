#ifndef __R1CS_EXAMPLES_CU__
#define __R1CS_EXAMPLES_CU__

#include "../../../../../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cuh"
#include "../../../../../depends/libstl-cuda/io.cuh"

#include <stdio.h>

namespace cuzk {

template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_with_field_input(const size_t num_constraints,
                                                            const size_t num_inputs, const FieldT& instance)
{
    // assert(num_inputs <= num_constraints + 2);

    libstl::vector<FieldT> full_variable_assignment(num_constraints + 2, instance);
    FieldT a = instance.random_element();
    FieldT b = instance.random_element();
    full_variable_assignment[0] = a;
    full_variable_assignment[1] = b;

    libstl::vector<r1cs_constraint<FieldT>> vconstraints(num_constraints);

    for (size_t i = 0; i < num_constraints-1; ++i)
    {
        linear_combination<FieldT>* pa = &(vconstraints[i].a);
        linear_combination<FieldT>* pb = &(vconstraints[i].b);
        linear_combination<FieldT>* pc = &(vconstraints[i].c);
        if (i % 2)
        {
            // a * b = c
            pa->add_term(i+1, 1, instance);
            pb->add_term(i+2, 1, instance);
            pc->add_term(i+3, 1, instance);
            FieldT tmp = a*b;
            full_variable_assignment[i+2] = tmp;
            a = b; b = tmp;
        }
        else
        {
            // a + b = c
            pb->add_term(0, 1, instance);
            pa->reserve_term(2, instance);
            pa->set_term(0, i+1, 1, instance);
            pa->set_term(1, i+2, 1, instance);
            pc->add_term(i+3, 1, instance);
            FieldT tmp = a+b;
            full_variable_assignment[i+2] = tmp;
            a = b; b = tmp;
        }
    }

    FieldT fin = instance.zero();
    vconstraints[num_constraints-1].a.reserve_term(1 + num_constraints, instance);
    vconstraints[num_constraints-1].b.reserve_term(1 + num_constraints, instance);
    for (size_t i = 1; i < 2 + num_constraints; ++i)
    {
        vconstraints[num_constraints-1].a.set_term(i-1, i, 1, instance);
        vconstraints[num_constraints-1].b.set_term(i-1, i, 1, instance);
        fin = fin + full_variable_assignment[i-1];
    }
    vconstraints[num_constraints-1].c.add_term(2 + num_constraints, 1, instance);
    full_variable_assignment[num_constraints + 1] = fin.squared();
    
    r1cs_constraint_system<FieldT> cs(libstl::move(vconstraints), num_inputs, 2 + num_constraints - num_inputs);

    libstl::vector<FieldT> primary_input(full_variable_assignment.begin(), full_variable_assignment.begin() + num_inputs);
    libstl::vector<FieldT> auxiliary_input(full_variable_assignment.begin() + num_inputs, full_variable_assignment.end());


    return r1cs_example<FieldT>(libstl::move(cs), libstl::move(primary_input), libstl::move(auxiliary_input));

}

template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_like_bellperson(const size_t num_constraints, const FieldT& instance)
{
    // assert(num_inputs <= num_constraints + 2);

    libstl::vector<FieldT> full_variable_assignment(num_constraints - 1, instance);;

    FieldT x_val = FieldT(instance.params, 2);

    full_variable_assignment[0] = x_val;

    libstl::vector<r1cs_constraint<FieldT>> vconstraints(num_constraints - 1);

    for (size_t i = 0; i < num_constraints - 2; ++i)
    {
        linear_combination<FieldT>* pa = &(vconstraints[i].a);
        linear_combination<FieldT>* pb = &(vconstraints[i].b);
        linear_combination<FieldT>* pc = &(vconstraints[i].c);

        // x_i * x_i = x_{i + 1}

        pa->add_term(i + 1, 1, instance);
        pb->add_term(i + 1, 1, instance);
        pc->add_term(i + 2, 1, instance);

        x_val = x_val * x_val;

        full_variable_assignment[i + 1] = x_val;
    }
    

    vconstraints[num_constraints - 2].a.add_term(0, x_val, instance);
    vconstraints[num_constraints - 2].b.add_term(0, 1, instance);
    vconstraints[num_constraints - 2].c.add_term(num_constraints - 1, 1, instance);

    // vconstraints[num_constraints - 2].a.add_term(0, 1, instance);
    // vconstraints[num_constraints - 2].b.add_term(0, 0, instance);
    // vconstraints[num_constraints - 2].c.add_term(0, 0, instance);

    r1cs_constraint_system<FieldT> cs(libstl::move(vconstraints), 0, num_constraints - 1);

    libstl::vector<FieldT> primary_input;
    libstl::vector<FieldT> auxiliary_input(full_variable_assignment);

    return r1cs_example<FieldT>(libstl::move(cs), libstl::move(primary_input), libstl::move(auxiliary_input));
}

template<typename FieldT>
__device__ FieldT read_fieldT(libstl::Reader& reader, const FieldT& instance)
{
    FieldT tmp;

    char hex[67];
    reader.read_str(hex);

    for (size_t k = 2; k < 66; k++)
        hex[k] = (hex[k] >= '0' && hex[k] <= '9') ? hex[k] - '0' : hex[k] - 'a' + 10; 

    tmp.mont_repr.clear();
    mpn_set_str_(tmp.mont_repr.data, (unsigned char*)hex + 2, 64, 16);

    // tmp.mont_repr.print();
    return FieldT(instance.params, tmp.mont_repr);
}

template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_from_file(char* r1cs_file, size_t length, const FieldT& instance)
{
    libstl::Reader reader(r1cs_file, length);

    size_t num_primary_inputs = reader.read_int();
    size_t num_auxiliary_inputs = reader.read_int();
    size_t num_constraints = reader.read_int();

    // printf("%llu %llu %llu\n", num_primary_inputs, num_auxiliary_inputs, num_constraints);

    libstl::vector<r1cs_constraint<FieldT>> vconstraints(num_constraints);

    for (size_t i = 0; i < num_constraints; i++)
    {
        size_t asize = reader.read_int();
        vconstraints[i].a.reserve_term(asize, instance);

        for (size_t j = 0; j < asize; j++)
        {
            FieldT coeff = read_fieldT(reader, instance);
            size_t index = reader.read_int();
            vconstraints[i].a.set_term(j, index, coeff, instance);
            // printf("%llu\n", index);
        }

        size_t bsize = reader.read_int();
        vconstraints[i].b.reserve_term(bsize, instance);

        for (size_t j = 0; j < bsize; j++)
        {
            FieldT coeff = read_fieldT(reader, instance);
            size_t index = reader.read_int();
            vconstraints[i].b.set_term(j, index, coeff, instance);
            // printf("%llu\n", index);
        }

        size_t csize = reader.read_int();
        vconstraints[i].c.reserve_term(csize, instance);

        for (size_t j = 0; j < csize; j++)
        {
            FieldT coeff = read_fieldT(reader, instance);
            size_t index = reader.read_int();
            vconstraints[i].c.set_term(j, index, coeff, instance);
            // printf("%llu\n", index);
        }

    }

    r1cs_constraint_system<FieldT> cs(libstl::move(vconstraints), num_primary_inputs, num_auxiliary_inputs);

    libstl::vector<FieldT> primary_input(num_primary_inputs);
    libstl::vector<FieldT> auxiliary_input(num_auxiliary_inputs);

    for (size_t i = 0; i < num_primary_inputs; i++) // read primary_assignment
        primary_input[i] = read_fieldT(reader, instance);

    for (size_t i = 0; i < num_auxiliary_inputs; i++) // read auxiliary_assignment
        auxiliary_input[i] = read_fieldT(reader, instance);

    printf("s %d\n", cs.is_satisfied(primary_input, auxiliary_input, instance));

    return r1cs_example<FieldT>(libstl::move(cs), libstl::move(primary_input), libstl::move(auxiliary_input));
}


}
#endif