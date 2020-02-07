// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/ManagedBuffer.hpp>

#include "linalg.hpp"

namespace vkt
{
    template <typename T>
    class Array3D : public ManagedBuffer<T>
    {
    public:
        typedef ManagedBuffer<T> Base;

        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        Array3D() = default;

        inline Array3D(Vec3i const& dims)
            : Base(dims.x * sizeof(dims.y) * dims.z * sizeof(T))
            , dims_(dims)
        {
        }

        Array3D(Array3D& rhs) = default;
        Array3D(Array3D&& rhs) = default;
       ~Array3D() = default;

        Array3D& operator=(Array3D& rhs) = default;
        Array3D& operator=(Array3D&& rhs) = default;

        inline iterator begin()
        {
            return Base::data_;
        }

        inline const_iterator begin() const
        {
            return Base::data_;
        }

        inline const_iterator cbegin()
        {
            return Base::data_;
        }

        inline iterator end()
        {
            return Base::data_ + numElements();
        }

        inline const_iterator end() const
        {
            return Base::data_ + numElements();
        }

        inline const_iterator cend()
        {
            return Base::data_ + numElements();
        }

        inline T& operator[](Vec3i const& index)
        {
            size_t linearIndex = index.z * dims_.x * std::size_t(dims_.y)
                               + index.y * dims_.x
                               + index.x;

            return data()[linearIndex];
        }

        inline T const& operator[](Vec3i const& index) const
        {
            size_t linearIndex = index.z * dims_.x * std::size_t(dims_.y)
                               + index.y * dims_.x
                               + index.x;

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
            const_cast<Array3D<T>*>(this)->migrate();

            return Base::data_;
        }

        inline Vec3i dims() const
        {
            return dims_;
        }

        inline std::size_t numElements() const
        {
            return dims_.x * dims_.y * dims_.z;
        }

    private:
        Vec3i dims_ = { 0, 0, 0 };

    };

} // vkt
