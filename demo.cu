// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include "pointer_adaptor.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

struct device_memory_accessor
{
  template<class T>
  __host__ __device__
  static T load(const T* ptr)
  {
#ifdef __CUDA_ARCH__
    return *ptr;
#else
    T result;
    if(cudaMemcpy(&result, ptr, sizeof(T), cudaMemcpyDefault) != cudaSuccess)
    {
      throw std::runtime_error("device_memory_accessor::load(): Error after cudaMemcpy");
    }

    return result;
#endif
  }

  // stores to a device pointer from an immediate value
  template<class T>
  __host__ __device__
  static void store(T* ptr, const T& value)
  {
#ifdef __CUDA_ARCH__
    *ptr = value;
#else
    if(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyDefault) != cudaSuccess)
    {
      throw std::runtime_error("device_memory_accessor::store(): Error after cudaMemcpy");
    }
#endif
  }

  // indirectly stores to a device pointer from another device pointer
  template<class T>
  __host__ __device__
  static void store(T* dst, const T* src)
  {
#ifdef __CUDA_ARCH__
    *dst = *src;
#else
    if(cudaMemcpy(dst, src, sizeof(T), cudaMemcpyDefault) != cudaSuccess)
    {
      throw std::runtime_error("device_memory_accessor::store(): Error after cudaMemcpy");
    }
#endif
  }
};

template<class T>
using my_device_ptr = pointer_adaptor<T, device_memory_accessor>;

template<class T>
using my_device_reference = typename pointer_adaptor<T, device_memory_accessor>::reference;

int main()
{
  thrust::device_vector<int> vec(4);
  thrust::sequence(vec.begin(), vec.end());

  thrust::device_ptr<int> d_ptr = vec.data();

  // construction from raw pointer
  my_device_ptr<int> ptr(vec.data().get());

  // test get
  for(int i = 0; i < 4; ++i)
  {
    assert((ptr + i).get() == (d_ptr + i).get());
  }

  // test deference
  for(int i = 0; i < 4; ++i)
  {
    assert(*(ptr + i) == *(d_ptr + i));
  }

  // test dereference-address-of-dererefence cycle
  for(int i = 0; i < 4; ++i)
  {
    my_device_reference<int> ref = *(ptr + i);

    my_device_ptr<int> ptr2 = &ref;

    assert(*ptr2 == *(d_ptr + i));
  }

  // test subscript
  for(int i = 0; i < 4; ++i)
  {
    assert(ptr[i] == d_ptr[i]);
  }

  // test direct store
  {
    for(int i = 0; i < 4; ++i)
    {
      ptr[i] = 3 - i;
    }

    for(int i = 0; i < 4; ++i)
    {
      assert(d_ptr[i] == 3 - i);
    }

    // restore state of vec
    thrust::sequence(vec.begin(), vec.end());
  }

  // test indirect store
  {
    for(int i = 0; i < 2; ++i)
    {
      ptr[i + 2] = ptr[i];
    }

    for(int i = 0; i < 4; ++i)
    {
      assert(d_ptr[i] == i % 2);
    }

    // restore state of vec
  }

  std::cout << "OK" << std::endl;
}

