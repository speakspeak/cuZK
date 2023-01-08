#ifndef __STL_CUDA_IO_CUH__
#define __STL_CUDA_IO_CUH__

namespace libstl {

class Reader {
public:

    __device__ Reader(char* buff, size_t length); 

    __device__ int read_int();
    
    __device__ void read_str(char *str);
   
    __device__ void skip();
   
private:
    char * buff;

    size_t index;

    size_t length;
};

}

#endif