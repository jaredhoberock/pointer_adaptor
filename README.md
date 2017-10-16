# pointer_adaptor
`pointer_adaptor` is a fancy pointer which adapts a handle to a value into a pointer-like type.

An `Accessor` object defines the mapping between a handle and its value by defining `.load()` and `.store()` functions. By default, the handle type is simply a raw pointer.

## Example

`pointer_adaptor` makes it easy to create fancy pointers which can point to remote address spaces. For example, we can use it to define a fancy pointer type which behaves similar to `thrust::device_ptr`:

```c++
#include "pointer_adaptor.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>

struct device_memory_accessor
{
  template<class T>
  __host__ __device__
  static T load(const T* ptr)
  {
    T result;
    if(cudaMemcpy(&result, ptr, sizeof(T), cudaMemcpyDefault) != cudaSuccess)
    {
      throw std::runtime_error("device_memory_accessor::load(): Error after cudaMemcpy");
    }

    return result;
  }

  // stores to a device pointer from an immediate value
  template<class T>
  __host__ __device__
  static void store(T* ptr, const T& value)
  {
    if(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyDefault) != cudaSuccess)
    {
      throw std::runtime_error("device_memory_accessor::store(): Error after cudaMemcpy");
    }
  }
};

template<class T>
using my_device_ptr = pointer_adaptor<T, device_memory_accessor>;

int main()
{
  thrust::device_vector<int> vec(4);
  thrust::sequence(vec.begin(), vec.end());

  thrust::device_ptr<int> d_ptr = vec.data();

  // alias vec's data through a my_device_ptr
  int* raw_ptr = vec.data().get();
  my_device_ptr<int> ptr(raw_ptr);

  // test get
  for(int i = 0; i < 4; ++i)
  {
    assert((ptr + i).get() == (d_ptr + i).get());
  }

  // test dereference
  for(int i = 0; i < 4; ++i)
  {
    assert(*(ptr + i) == *(d_ptr + i));
  }

  // test subscript
  for(int i = 0; i < 4; ++i)
  {
    assert(ptr[i] == d_ptr[i]);
  }

  std::cout << "OK" << std::endl;
}
```

