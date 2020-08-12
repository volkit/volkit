// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/ManagedBuffer.hpp>

#include "linalg.hpp"

namespace vkt
{
    template <typename T>
    class Array2D : public ManagedBuffer<T>
    {
    public:
        typedef ManagedBuffer<T> Base;

        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        Array2D() = default;

        inline Array2D(Vec2i const& dims)
            : Base(dims.x * std::size_t(dims.y) * sizeof(T))
            , dims_(dims)
        {
        }

        Array2D(Array2D& rhs) = default;
        Array2D(Array2D&& rhs) = default;
       ~Array2D() = default;

        Array2D& operator=(Array2D& rhs) = default;
        Array2D& operator=(Array2D&& rhs) = default;

        inline void resize(Vec2i const& dims)
        {
            Base::resize(dims.x * std::size_t(dims.y) * sizeof(T));
            dims_ = dims;
        }

        inline void fill(T& value)
        {
            Base::fill(value);
        }

        inline void fill(T const& value)
        {
            Base::fill(value);
        }

        inline iterator begin()
        {
            return data();
        }

        inline const_iterator begin() const
        {
            return data();
        }

        inline const_iterator cbegin()
        {
            return data();
        }

        inline iterator end()
        {
            return data() + numElements();
        }

        inline const_iterator end() const
        {
            return data() + numElements();
        }

        inline const_iterator cend()
        {
            return data() + numElements();
        }

        inline T& operator[](Vec2i const& index)
        {
            size_t linearIndex = index.y * std::size_t(dims_.x) + index.x;

            return data()[linearIndex];
        }

        inline T const& operator[](Vec2i const& index) const
        {
            size_t linearIndex = index.y * std::size_t(dims_.x) + index.x;

            return data()[linearIndex];
        }

        inline bool empty() const
        {
            return numElements() == 0;
        }

        inline T* data()
        {
            Base::migrate();

            return Base::data_;
        }

        inline T const* data() const
        {
            const_cast<Array2D<T>*>(this)->migrate();

            return Base::data_;
        }

        inline Vec2i dims() const
        {
            return dims_;
        }

        inline std::size_t numElements() const
        {
            return dims_.x * std::size_t(dims_.y);
        }

    private:
        Vec2i dims_ = { 0, 0 };

    };

} // vkt
