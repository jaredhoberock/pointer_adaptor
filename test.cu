#include "pointer_adaptor.hpp"
#include <iostream>
#include <cassert>

struct empty_accessor {};

int main()
{
  {
    // test default construction
    pointer_adaptor<int, empty_accessor> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    pointer_adaptor<int, empty_accessor> ptr(nullptr);

    assert(ptr.get() == nullptr);
    assert(!ptr);
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    pointer_adaptor<int, empty_accessor> ptr(array);

    // test get
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).get() == &array[i]);
    }

    // test deference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == array[i]);
    }

    // test store
    for(int i = 0; i < 4; ++i)
    {
      ptr[i] = 4 - i;
    }

    for(int i = 0; i < 4; ++i)
    {
      assert(array[i] == 4 - i);
    }
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    pointer_adaptor<const int, empty_accessor> ptr(array);

    // test get
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).get() == &array[i]);
    }

    // test deference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == array[i]);
    }
  }

  std::cout << "OK" << std::endl;
}

