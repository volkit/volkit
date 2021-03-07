// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include "macros.hpp"

namespace vkt
{
    template <typename T>
    class ArrayView1D
    {
    public:
        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        ArrayView1D() = default;

        ArrayView1D(T* data, std::size_t dims)
            : data_(data)
            , dims_(dims)
        {
        }

        VKT_FUNC
        inline iterator begin()
        {
            return data();
        }

        VKT_FUNC
        inline const_iterator begin() const
        {
            return data();
        }

        VKT_FUNC
        inline const_iterator cbegin() const
        {
            return data();
        }

        VKT_FUNC
        inline iterator end()
        {
            return data() + numElements();
        }

        VKT_FUNC
        inline const_iterator end() const
        {
            return data() + numElements();
        }

        VKT_FUNC
        inline const_iterator cend() const
        {
            return data() + numElements();
        }

        VKT_FUNC
        inline T& operator[](std::size_t index)
        {
            return data()[index];
        }

        VKT_FUNC
        inline T const& operator[](std::size_t index) const
        {
            return data()[index];
        }

        VKT_FUNC
        inline bool empty() const
        {
            return numElements() == 0;
        }

        VKT_FUNC
        inline T* data()
        {
            return data_;
        }

        VKT_FUNC
        inline T const* data() const
        {
            return data_;
        }

        VKT_FUNC
        inline T const* cdata() const
        {
            return data_;
        }

        VKT_FUNC
        inline std::size_t dims() const
        {
            return dims_;
        }

        VKT_FUNC
        inline std::size_t numElements() const
        {
            return dims_;
        }

    private:
        T* data_;
        std::size_t dims_;

    };
} // vkt
