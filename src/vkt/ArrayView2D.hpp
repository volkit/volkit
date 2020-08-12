// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include "linalg.hpp"
#include "macros.hpp"

namespace vkt
{
    template <typename T>
    class ArrayView2D
    {
    public:
        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        ArrayView2D() = default;

        ArrayView2D(T* data, Vec2i const& dims)
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
        inline T& operator[](Vec2i const& index)
        {
            size_t linearIndex = index.y * std::size_t(dims_.x) + index.x;

            return data()[linearIndex];
        }

        VKT_FUNC
        inline T const& operator[](Vec2i const& index) const
        {
            size_t linearIndex = index.y * std::size_t(dims_.x) + index.x;

            return data()[linearIndex];
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
        inline Vec2i dims() const
        {
            return dims_;
        }

        VKT_FUNC
        inline std::size_t numElements() const
        {
            return dims_.x * std::size_t(dims_.y);
        }

    private:
        T* data_;
        Vec2i dims_;

    };
} // vkt
