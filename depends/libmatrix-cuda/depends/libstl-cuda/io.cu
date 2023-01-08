#ifndef __STL_CUDA_IO_CU__
#define __STL_CUDA_IO_CU__

#include "io.cuh"

namespace libstl {

__device__ Reader::Reader(char* buff, size_t length) : buff(buff), index(0), length(length) 
{

}

__device__ int Reader::read_int()
{
    skip();

    int res = 0;

    while (index < length && buff[index] >= '0' && buff[index] <= '9') 
        res = 10 * res + (buff[index++] - '0');

    return res;
}


__device__ void Reader::read_str(char *str)
{
    skip();

    size_t sindex = 0;

    while (index < length && buff[index] != ' ' && buff[index] != '\n'&& buff[index] != '\r' )
        str[sindex++] = buff[index++];

    str[sindex] = '\0';
}

__device__ void Reader::skip()
{
    while (index < length && (buff[index] == ' ' || buff[index] == '\n'|| buff[index] == '\r' ))
        ++index;
}

}

#endif