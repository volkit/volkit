#pragma once

#include <cstddef>

#include <vkt/ManagedBuffer.hpp>

#include "linalg.hpp"

namespace vkt
{
    template <typename T>
    class Array3D : public ManagedBuffer
    {
    public:
        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        Array3D() = default;

        inline Array3D(vec3i const& dims)
            : dims_(dims)
        {
            ManagedBuffer::allocate(numElements() * sizeof(T));
        }

        inline Array3D(Array3D& rhs)
            : dims_(rhs.dims_)
        {
            ManagedBuffer::allocate(numElements() * sizeof(T));

            ManagedBuffer::copy((ManagedBuffer&)rhs);
        }

        inline ~Array3D()
        {
            ManagedBuffer::deallocate();
        }

        inline Array3D& operator=(Array3D& rhs)
        {
            if (&rhs != this)
            {
                dims_ = rhs.dims_;

                ManagedBuffer::deallocate();

                ManagedBuffer::allocate(numElements() * sizeof(T));

                ManagedBuffer::copy((ManagedBuffer&)rhs);
            }

            return *this;
        }

        inline iterator begin()
        {
            return reinterpret_cast<T*>(ManagedBuffer::data_);
        }

        inline const_iterator begin() const
        {
            return reinterpret_cast<T const*>(ManagedBuffer::data_);
        }

        inline const_iterator cbegin()
        {
            return reinterpret_cast<T const*>(ManagedBuffer::data_);
        }

        inline iterator end()
        {
            return reinterpret_cast<T*>(ManagedBuffer::data_) + numElements();
        }

        inline const_iterator end() const
        {
            return reinterpret_cast<T const*>(ManagedBuffer::data_) + numElements();
        }

        inline const_iterator cend()
        {
            return reinterpret_cast<T const*>(ManagedBuffer::data_) + numElements();
        }

        inline T& operator[](vec3i const& index)
        {
            size_t linearIndex = index.z * dims_.x * std::size_t(dims_.y)
                               + index.y * dims_.x
                               + index.x;

            return data()[linearIndex];
        }

        inline T const& operator[](vec3i const& index) const
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
            migrate();

            return reinterpret_cast<T*>(ManagedBuffer::data_);
        }

        inline T const* data() const
        {
            const_cast<Array3D<T>*>(this)->migrate();

            return reinterpret_cast<T const*>(ManagedBuffer::data_);
        }

        inline vec3i dims() const
        {
            return dims_;
        }

        inline std::size_t numElements() const
        {
            return dims_.x * dims_.y * dims_.z;
        }

    private:
        vec3i dims_ = vec3i(0);

    };

} // vkt
