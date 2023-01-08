#ifndef __STL_CUDA_VECTOR_CU__
#define __STL_CUDA_VECTOR_CU__

#include "utility.cuh"

namespace libstl {

template<typename T>
__host__ __device__ vector<T>::vector() : _size(0), _data(nullptr)
{

}

template<typename T>
__device__ vector<T>::vector(size_t n) : _size(n)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));

    uninitialized_fill_n(_data, _size, T());
}

template<typename T>
__device__ vector<T>::vector(size_t n, size_t gridSize, size_t blockSize) : _size(n)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));

    T* instance = libstl::create<T>(T());
    p_uninitialized_fill_n(_data, _size, *instance, gridSize, blockSize);
}


template<typename T>
__device__ vector<T>::vector(size_t n, const T& val) : _size(n)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    uninitialized_fill_n(_data, _size, val);
}


template<typename T>
__device__ vector<T>::vector(size_t n, const T& val, size_t gridSize, size_t blockSize) : _size(n)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    p_uninitialized_fill_n(_data, _size, val, gridSize, blockSize);
}


template<typename T>
__device__ vector<T>::vector(const self_type& x) : _size(x._size)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    uninitialized_copy_n(x.begin(), _size, _data);
}

template<typename T>
__device__ vector<T>::vector(const self_type& x, size_t gridSize, size_t blockSize) : _size(x._size)
{
    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    p_uninitialized_copy_n(x.begin(), _size, _data, gridSize, blockSize);
}

template<typename T>
__device__ vector<T>::vector(self_type&& x) : _size(x._size), _data(x._data)
{
    x._size = 0;
    x._data = nullptr;
}

template<typename T>
__device__ vector<T>::vector(const_iterator first, const_iterator last) : _size(last - first)
{
    iterator nfirst = (iterator)first;
    iterator nlast = (iterator)last;

    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    uninitialized_copy(nfirst, nlast, _data);
}

template<typename T>
__device__ vector<T>::vector(const_iterator first, const_iterator last, size_t gridSize, size_t blockSize) : _size(last - first)
{
    iterator nfirst = (iterator)first;
    iterator nlast = (iterator)last;

    _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    p_uninitialized_copy(nfirst, nlast, _data, gridSize, blockSize);
}
    
template<typename T>
__host__ __device__ vector<T>::~vector()
{
}

template<typename T>
__device__ vector<T>& vector<T>::operator= (const self_type& x)
{
    if (this == &x)
        return *this;

    if (_size != x._size)
    {
        _size = x._size;
        _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    }

    uninitialized_copy_n(x.begin(), _size, _data);

    return *this;
}

template<typename T>
__device__ vector<T>& vector<T>::operator=(self_type&& x)
{
    if (this == &x)
        return *this;

    _size = x._size;
    _data = x._data;

    x._size = 0;
    x._data = nullptr;

    return *this;
}

template<typename T>
__device__ void vector<T>::pcopy (const self_type& x, size_t gridSize, size_t blockSize)
{
    if (this == &x)
        return ;

    if (_size != x._size)
    {
        _size = x._size;
        _data = _size == 0 ? nullptr : (T*)_allocManager.allocate(_size * sizeof(T));
    }

    p_uninitialized_copy_n(x.begin(), _size, _data, gridSize, blockSize);

    return;
}

template<typename T>
__host__ void vector<T>::pcopy_host(self_type& x, size_t gridSize, size_t blockSize)
{
    if (this == &x)
        return ;

    size_t my_vector_size;
    size_t x_vector_size;
    T* x_vector_addr;
    libstl::get_host(&my_vector_size, &this->_size);
    libstl::get_host(&x_vector_size, &x._size);
    libstl::get_host(&x_vector_addr, &x._data);

    if (my_vector_size != x_vector_size)
    {
        T* temp = x_vector_size == 0 ? nullptr : (T*)_allocManager.allocate_host(x_vector_size * sizeof(T));
        p_uninitialized_copy_n_host(x_vector_addr, x_vector_size, temp, gridSize, blockSize);

        send_host(&this->_size , &x_vector_size);
        send_host(&this->_data, &temp);
    }

    return;
}

template<typename T>
__host__ __device__ T& vector<T>::operator[] (size_t n)
{
    return _data[n];
}

template<typename T>
__host__ __device__ const T& vector<T>::operator[] (size_t n) const
{
    return _data[n];
}

template<typename T>
__device__ T& vector<T>::front()
{
    return _data[0];
}

template<typename T>
__device__ const T& vector<T>::front() const
{
    return _data[0];
}

template<typename T>
__device__ T& vector<T>::back()
{
    return _data[_size - 1];
}

template<typename T>
__device__ const T& vector<T>::back() const
{
    return _data[_size - 1];
}

template<typename T>
__device__ vector<T>::iterator vector<T>::begin()
{
    return _data;
}

template<typename T>
__device__ vector<T>::const_iterator vector<T>::begin() const
{
    return _data;
}

template<typename T>
__device__ vector<T>::const_iterator vector<T>::cbegin() const
{
    return _data;
}

template<typename T>
__device__ vector<T>::iterator vector<T>::end()
{
    return _data + _size;
}

template<typename T>
__device__ vector<T>::const_iterator vector<T>::end() const
{
    return _data + _size;
}

template<typename T>
__device__ vector<T>::const_iterator vector<T>::cend() const
{
    return _data + _size;
}

template<typename T>
__device__ size_t vector<T>::size() const
{
    return _size;
}

template<typename T>
__host__ size_t vector<T>::size_host()
{
    size_t size;
    get_host(&size, &this->_size);
    return size;
}

template<typename T>
__device__ void vector<T>::resize(size_t n)
{
    if (n == _size)
        return;

    T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate(n * sizeof(T));

    if (n > _size)
    {
        uninitialized_move_n(_data, _size, temp);
        uninitialized_fill_n(temp + _size, n - _size, T());
    }
    else
        uninitialized_move_n(_data, n, temp);

    _size = n;
    _data = temp;
}

template<typename T>
__host__ void vector<T>::resize_host(size_t n)
{
    if (n == _size)
        return;

    T* temp = nullptr;
    if(n != 0)
        if(cudaHostAlloc((void **)&temp, n * sizeof(T), cudaHostAllocDefault) != cudaSuccess) printf("host alloc error\n");

    if (n > _size)
    {
        uninitialized_move_n(_data, _size, temp);
        uninitialized_fill_n(temp + _size, n - _size, T());
    }
    else
        uninitialized_move_n(_data, n, temp);

    _size = n;
    _data = temp;
}


template<typename T>
__device__ void vector<T>::presize(size_t n, size_t gridSize, size_t blockSize)
{
    if (n != _size)
    {
        T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate(n * sizeof(T));
        T* instance = libstl::create<T>(T());

        if (n > _size)
        {
            p_uninitialized_move_n(_data, _size, temp, gridSize, blockSize);
            p_uninitialized_fill_n(temp + _size, n - _size, *instance, gridSize, blockSize);
        }
        else
            p_uninitialized_move_n(_data, n, temp, gridSize, blockSize);

        _size = n;
        _data = temp;
    }
}


template<typename T>
__host__ void vector<T>::presize_host(size_t n, size_t gridSize, size_t blockSize)
{
    size_t vector_size;
    T* vector_addr;
    libstl::get_host(&vector_size, &this->_size);
    libstl::get_host(&vector_addr, &this->_data);

    if (n != vector_size)
    {
        T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate_host(n * sizeof(T));
        T* instance = create_host<T>();

        if (n > vector_size)
        {
            p_uninitialized_move_n_host((T*)vector_addr, vector_size, temp, gridSize, blockSize);
            p_uninitialized_fill_n_host((T*)temp + vector_size, n - vector_size, instance, gridSize, blockSize);
        }
        else
            p_uninitialized_move_n_host((T*)vector_addr, n, temp, gridSize, blockSize);

        send_host(&this->_size , &n);
        send_host(&this->_data , &temp);
    }
}

template<typename T>
__device__ void vector<T>::resize(size_t n, const T& val)
{
    if (n == _size)
        return;

    T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate(n * sizeof(T));

    if (n > _size)
    {
        uninitialized_move_n(_data, _size, temp);
        uninitialized_fill_n(temp + _size, n - _size, val);
    }
    else
        uninitialized_move_n(_data, n, temp);

    _size = n;
    _data = temp;
}



template<typename T>
__device__ void vector<T>::presize(size_t n, const T& val, size_t gridSize, size_t blockSize)
{
    if (n != _size)
    {
        T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate(n * sizeof(T));

        if (n > _size)
        {
            p_uninitialized_move_n(_data, _size, temp, gridSize, blockSize);
            p_uninitialized_fill_n(temp + _size, n - _size, val, gridSize, blockSize);
        }
        else
            p_uninitialized_move_n(_data, n, temp, gridSize, blockSize);

        _size = n;
        _data = temp;
    }
}

template<typename T>
__host__ void vector<T>::presize_host(size_t n, const T* val, size_t gridSize, size_t blockSize)
{
    size_t vector_size;
    T* vector_addr;
    libstl::get_host(&vector_size, &this->_size);
    libstl::get_host(&vector_addr, &this->_data);

    if (n != vector_size)
    {
        T* temp = n == 0 ? nullptr : (T*)_allocManager.allocate_host(n * sizeof(T));

        if (n > vector_size)
        {
            p_uninitialized_move_n_host((T*)vector_addr, vector_size, temp, gridSize, blockSize);
            p_uninitialized_fill_n_host((T*)temp + vector_size, n - vector_size, val, gridSize, blockSize);
        }
        else
            p_uninitialized_move_n_host((T*)vector_addr, n, temp, gridSize, blockSize);
        send_host(&this->_size , &n);
        send_host(&this->_data , &temp);
    }
}


template<typename T>
__device__ size_t vector<T>::memory_need(size_t n)
{
    return n * sizeof(T);
}


template<typename T, typename H>
__host__ void vector_device2host(libstl::vector<H>* hv, const libstl::vector<T>* dv, cudaStream_t stream)
{
    size_t vector_size;
    void* vector_addr;
    cudaMemcpy(&vector_size, &dv->_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vector_addr, &dv->_data, sizeof(void *), cudaMemcpyDeviceToHost);
    hv->resize_host(vector_size);

    if(stream == 0)
        cudaMemcpy(hv->_data, (void *)vector_addr, vector_size * sizeof(H), cudaMemcpyDeviceToHost);
    else
        cudaMemcpyAsync(hv->_data, (void *)vector_addr, vector_size * sizeof(H), cudaMemcpyDeviceToHost, stream);
}

template<typename T, typename H>
__host__ void vector_host2device(libstl::vector<T>* dv, const libstl::vector<H>* hv, cudaStream_t stream)
{
    size_t vector_size = hv->_size;
    dv->presize_host(vector_size, 512, 32);
    void* vector_addr;
    cudaMemcpy(&vector_addr, &dv->_data, sizeof(void *), cudaMemcpyDeviceToHost);
    if(stream == 0)
        cudaMemcpy((void *)vector_addr, hv->_data, vector_size * sizeof(T), cudaMemcpyHostToDevice);
    else
        cudaMemcpyAsync((void *)vector_addr, hv->_data, vector_size * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
__host__ void vector_device2host(libstl::vector<T>* hv, const libstl::vector<T>* dv, cudaStream_t stream)
{
    vector_device2host<T, T>(hv, dv, stream);
}

template<typename T>
__host__ void vector_host2device(libstl::vector<T>* dv, const libstl::vector<T>* hv, cudaStream_t stream)
{
    vector_host2device<T, T>(dv, hv, stream);
}



}

#endif