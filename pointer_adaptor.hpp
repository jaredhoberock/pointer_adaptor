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

#pragma once

#include "is_detected.hpp"
#include <utility>

#define __POINTER_ADAPTOR_CONCATENATE_IMPL(x, y) x##y

#define __POINTER_ADAPTOR_CONCATENATE(x, y) __POINTER_ADAPTOR_CONCATENATE_IMPL(x, y)

#define __POINTER_ADAPTOR_MAKE_UNIQUE(x) __POINTER_ADAPTOR_CONCATENATE(x, __COUNTER__)

#define __POINTER_ADAPTOR_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#define __POINTER_ADAPTOR_REQUIRES(...) __POINTER_ADAPTOR_REQUIRES_IMPL(__POINTER_ADAPTOR_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)


template<class T, class Accessor>
class pointer_adaptor : private Accessor
{
  private:
    // note that we derive from Accessor for the empty base class optimization
    using super_t = Accessor;

    template<class U>
    using member_handle_type = typename U::handle_type;

    template<class U>
    using member_difference_type = typename U::difference_type;

  public:
    using element_type = T;
    using accessor_type = Accessor;
    using handle_type = detected_or_t<T*, member_handle_type, Accessor>;
    using difference_type = detected_or_t<std::ptrdiff_t, member_difference_type, Accessor>;

    class reference : private Accessor
    {
      private:
        using super_t = Accessor;

      public:
        reference() = default;

        reference(const reference&) = default;

        reference(const handle_type& handle, const accessor_type& accessor)
          : super_t(accessor), handle_(handle)
        {}

        operator element_type () const
        {
          return this->load(accessor(), handle_);
        }

        template<__POINTER_ADAPTOR_REQUIRES(
                  std::is_assignable<element_type&, const element_type&>::value
                )>
        reference operator=(const element_type& value) const
        {
          this->store(accessor(), handle_, value);
          return *this;
        }

        // this overload simply generates a diagnostic with the static_assert
        template<__POINTER_ADAPTOR_REQUIRES(
                  !std::is_assignable<element_type&, const element_type&>::value
                )>
        reference operator=(const element_type&) const
        {
          static_assert(std::is_assignable<element_type&, const element_type&>::value, "pointer_adaptor element_type is not assignable.");
          return *this;
        }

      private:
        const accessor_type& accessor() const
        {
          return *this;
        }

        accessor_type& accessor()
        {
          return *this;
        }

        template<class U>
        using member_load_t = decltype(std::declval<U>().load(std::declval<handle_type>()));

        template<class U>
        using has_member_load = is_detected_exact<element_type, member_load_t, accessor_type>; 

        template<__POINTER_ADAPTOR_REQUIRES(has_member_load<accessor_type>::value)>
        static element_type load(const accessor_type& accessor, const handle_type& handle)
        {
          return accessor.load(handle);
        }

        template<__POINTER_ADAPTOR_REQUIRES(!has_member_load<accessor_type>::value and std::is_pointer<handle_type>::value)>
        static element_type load(const accessor_type& accessor, const handle_type& handle)
        {
          return *handle;
        }


        template<class U>
        using member_store_t = decltype(std::declval<U>().store(std::declval<handle_type>(), std::declval<element_type>()));

        template<class U>
        using has_member_store = is_detected<member_store_t, accessor_type>;

        template<__POINTER_ADAPTOR_REQUIRES(has_member_store<accessor_type>::value)>
        static void store(const accessor_type& accessor, const handle_type& handle, const element_type& value)
        {
          return accessor.store(handle, value);
        }

        template<__POINTER_ADAPTOR_REQUIRES(!has_member_store<accessor_type>::value and std::is_pointer<handle_type>::value)>
        static void store(const accessor_type&, const handle_type& handle, const element_type& value)
        {
          *handle = value;
        }

        handle_type handle_;
    };

    pointer_adaptor() = default;

  private:
    template<class U>
    using member_null_handle_t = decltype(std::declval<U>().null_handle());

    template<class U>
    using has_member_null_handle = is_detected_exact<handle_type, member_null_handle_t, accessor_type>;

    template<__POINTER_ADAPTOR_REQUIRES(has_member_null_handle<accessor_type>::value)>
    static handle_type null_handle(const accessor_type& accessor)
    {
      return accessor.null_handle();
    }

    template<__POINTER_ADAPTOR_REQUIRES(!has_member_null_handle<accessor_type>::value && std::is_constructible<handle_type, std::nullptr_t>::value)>
    static handle_type null_handle(const accessor_type&)
    {
      return nullptr;
    }

  public:
    pointer_adaptor(std::nullptr_t) noexcept
      : pointer_adaptor(null_handle(accessor_type()))
    {}

    pointer_adaptor(const handle_type& h) noexcept
      : pointer_adaptor(h, accessor_type())
    {}

    pointer_adaptor(const handle_type& h, const accessor_type& a) noexcept
      : super_t(a), handle_(h)
    {}

    pointer_adaptor(const handle_type& h, accessor_type&& a) noexcept
      : super_t(std::move(a)), handle_(h)
    {}

    template<class U, class OtherAccessor,
             __POINTER_ADAPTOR_REQUIRES(
               std::is_convertible<U*, T*>::value // This first requirement disables unreasonable conversions, e.g. pointer_adaptor<void,...> -> pointer_adaptor<int,...>
               and std::is_convertible<typename U::handle_type, handle_type>::value
               and std::is_convertible<OtherAccessor, accessor_type>::value
            )>
    pointer_adaptor(const pointer_adaptor<U,OtherAccessor>& other)
      : pointer_adaptor(other.get(), other.accessor())
    {}

    // returns the underlying handle
    const handle_type& get() const noexcept
    {
      return handle_;
    }

    // returns the accessor
    Accessor& accessor() noexcept
    {
      return *this;
    }

    // returns the accessor
    const Accessor& accessor() const noexcept
    {
      return *this;
    }

    // conversion to bool
    explicit operator bool() const noexcept
    {
      return get() != null_handle(accessor());
    }

    // dereference
    reference operator*() const
    {
      return reference(get(), accessor());
    }

    // subscript
    reference operator[](difference_type i) const
    {
      return *(*this + i);
    }

    // pre-increment
    pointer_adaptor operator++()
    {
      this->advance(accessor(), handle_, 1);
      return *this;
    }

    // pre-decrement
    pointer_adaptor operator--()
    {
      this->advance(accessor(), handle_, -1);
      return *this;
    }

    // post-increment
    pointer_adaptor operator++(int)
    {
      pointer_adaptor result = *this;
      operator++();
      return result;
    }

    // post-decrement
    pointer_adaptor operator--(int)
    {
      pointer_adaptor result = *this;
      operator--();
      return result;
    }

    // plus
    pointer_adaptor operator+(difference_type n) const
    {
      pointer_adaptor result = *this;
      result += n;
      return result;
    }

    // minus
    pointer_adaptor operator-(difference_type n) const
    {
      pointer_adaptor result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    pointer_adaptor& operator+=(difference_type n)
    {
      this->advance(accessor(), handle_, n);
      return *this;
    }

    // minus-equal
    pointer_adaptor& operator-=(difference_type n)
    {
      this->advance(accessor(), handle_, -n);
      return *this;
    }

  private:
    template<class U>
    using member_advance_t = decltype(std::declval<U>().advance(std::declval<handle_type>(), std::declval<difference_type>()));

    template<class U>
    using has_member_advance = is_detected<member_advance_t, accessor_type>; 

    template<__POINTER_ADAPTOR_REQUIRES(has_member_advance<accessor_type>::value)>
    static void advance(accessor_type& accessor, handle_type& handle, difference_type n)
    {
      accessor.advance(handle, n);
    }

    template<__POINTER_ADAPTOR_REQUIRES(!has_member_advance<accessor_type>::value && std::is_pointer<handle_type>::value)>
    static void advance(accessor_type& accessor, handle_type& handle, difference_type n)
    {
      handle += n;
    }

    handle_type handle_;
};

