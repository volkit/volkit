// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/ManagedBuffer.hpp>

namespace vkt
{
    template <typename T>
    class Array1D : public ManagedBuffer<T>
    {
    public:
        typedef ManagedBuffer<T> Base;

        typedef T value_type;
        typedef T* iterator;
        typedef T const* const_iterator;

    public:
        Array1D() = default;

        inline Array1D(std::size_t dims)
            : Base(dims * sizeof(T))
            , dims_(dims)
        {
        }

        Array1D(Array1D& rhs) = default;
        Array1D(Array1D&& rhs) = default;
       ~Array1D() = default;

        Array1D& operator=(Array1D& rhs) = default;
        Array1D& operator=(Array1D&& rhs) = default;

        inline void resize(std::size_t dims)
        {
            Base::resize(dims * sizeof(T));
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

        inline T& operator[](std::size_t index)
        {
            return data()[index];
        }

        inline T const& operator[](std::size_t index) const
        {
            return data()[index];
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
            const_cast<Array1D<T>*>(this)->migrate();

            return Base::data_;
        }

        inline std::size_t dims() const
        {
            return dims_;
        }

        inline std::size_t numElements() const
        {
            return dims_;
        }

    private:
        std::size_t dims_ = 0;

    };

} // vkt
