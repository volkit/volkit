// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/ManagedResource.hpp>
#include <vkt/Memory.hpp>

namespace vkt
{

    /*!
     * @brief  a buffer whose memory can be migrated between devices on demand
     */

    template <typename T>
    class ManagedBuffer
    {
    public:
        typedef T value_type;

    public:
        ManagedBuffer(std::size_t size = 0);
        ManagedBuffer(ManagedBuffer& rhs);
        ManagedBuffer(ManagedBuffer&& rhs);
        virtual ~ManagedBuffer();

        ManagedBuffer& operator=(ManagedBuffer& rhs);
        ManagedBuffer& operator=(ManagedBuffer&& rhs);

        ResourceHandle getResourceHandle() const;

        /*!
         * @brief  if device changed, copy data arrays into new address space
         */
        void migrate();

    protected:
        void allocate(std::size_t size);
        void deallocate();
        void resize(std::size_t size);
        void fill(T& value);
        void fill(T const& value);
        void copy(ManagedBuffer& rhs);

        T* data_ = nullptr;
        std::size_t size_ = 0;

    private:
        ExecutionPolicy lastAllocationPolicy_ = {};

        ResourceHandle resourceHandle_ = ResourceHandle(-1);

    };




    template <typename T>
    ManagedBuffer<T>::ManagedBuffer(std::size_t size)
        : data_(nullptr)
        , size_(size)
    {
        lastAllocationPolicy_ = {};

        allocate(size);

        resourceHandle_ = RegisterManagedResource(this);
    }

    template <typename T>
    ManagedBuffer<T>::ManagedBuffer(ManagedBuffer& rhs)
        : data_(nullptr)
        , size_(rhs.size_)
        , lastAllocationPolicy_(rhs.lastAllocationPolicy_)
    {
        rhs.migrate();

        allocate(rhs.size_);

        copy(rhs);

        resourceHandle_ = RegisterManagedResource(this);
    }

    template <typename T>
    ManagedBuffer<T>::ManagedBuffer(ManagedBuffer&& rhs)
        : data_(nullptr)
        , size_(rhs.size_)
        , lastAllocationPolicy_(rhs.lastAllocationPolicy_)
    {
        rhs.migrate();

        allocate(rhs.size_);

        copy(rhs);

        resourceHandle_ = RegisterManagedResource(this);

        rhs.deallocate();
        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }

    template <typename T>
    ManagedBuffer<T>::~ManagedBuffer()
    {
        UnregisterManagedResource(resourceHandle_);

        deallocate();
    }

    template <typename T>
    ManagedBuffer<T>& ManagedBuffer<T>::operator=(ManagedBuffer& rhs)
    {
        if (&rhs != this)
        {
            rhs.migrate();

            size_ = rhs.size_;
            lastAllocationPolicy_ = rhs.lastAllocationPolicy_;

            deallocate();

            allocate(rhs.size_);

            copy(rhs);
        }

        return *this;
    }

    template <typename T>
    ManagedBuffer<T>& ManagedBuffer<T>::operator=(ManagedBuffer&& rhs)
    {
        if (&rhs != this)
        {
            rhs.migrate();

            size_ = rhs.size_;
            lastAllocationPolicy_ = rhs.lastAllocationPolicy_;

            deallocate();

            allocate(rhs.size_);

            copy(rhs);

            rhs.deallocate();
            rhs.data_ = nullptr;
            rhs.size_ = 0;
        }

        return *this;
    }

    template <typename T>
    ResourceHandle ManagedBuffer<T>::getResourceHandle() const
    {
        return resourceHandle_;
    }

    template <typename T>
    void ManagedBuffer<T>::migrate()
    {
        ExecutionPolicy ep = GetThreadExecutionPolicy();

        if (ep.device != lastAllocationPolicy_.device)
        {
            T* newData = nullptr;

            Allocate((void**)&newData, size_ * sizeof(T));

            CopyKind ck = ep.device == ExecutionPolicy::Device::GPU
                            ? CopyKind::HostToDevice
                            : CopyKind::DeviceToHost;

            Memcpy(newData, data_, size_ * sizeof(T), ck);

            // Free the data array with the old allocation policy
            SetThreadExecutionPolicy(lastAllocationPolicy_);

            Free(data_);

            // Restore the most recent execution policy
            SetThreadExecutionPolicy(ep);

            // This is now also the policy we used for the last allocation
            lastAllocationPolicy_ = ep;

            // Migration complete
            data_ = newData;
        }
    }

    template <typename T>
    void ManagedBuffer<T>::allocate(std::size_t size)
    {
        lastAllocationPolicy_ = GetThreadExecutionPolicy();

        size_ = size;

        Allocate((void**)&data_, size_ * sizeof(T));
    }

    template <typename T>
    void ManagedBuffer<T>::deallocate()
    {
        // Free with last allocation policy
        ExecutionPolicy curr = GetThreadExecutionPolicy();

        SetThreadExecutionPolicy(lastAllocationPolicy_);

        Free(data_);

        // Make policy from before Free() call current
        lastAllocationPolicy_ = curr;
        SetThreadExecutionPolicy(curr);
    }

    template <typename T>
    void ManagedBuffer<T>::resize(std::size_t size)
    {
        std::size_t oldSize = size_;
        std::size_t newSize = size;

        migrate();

        ExecutionPolicy ep = GetThreadExecutionPolicy();

        CopyKind ck = ep.device == ExecutionPolicy::Device::GPU
                        ? CopyKind::DeviceToDevice
                        : CopyKind::HostToHost;

        T* temp = nullptr;
        Allocate((void**)&temp, oldSize * sizeof(T));
        Memcpy(temp, data_, std::min(oldSize * sizeof(T), newSize * sizeof(T)), ck);

        Free(data_);
        Allocate((void**)&data_, newSize * sizeof(T));
        Memcpy(data_, temp, std::min(oldSize * sizeof(T), newSize * sizeof(T)), ck);

        Free(temp);

        size_ = newSize;
    }

    template <typename T>
    void ManagedBuffer<T>::fill(T& value)
    {
        migrate();

        T* buf = new T[size_];
        for (std::size_t i = 0; i < size_; ++i)
            buf[i] = value;

        ExecutionPolicy ep = GetThreadExecutionPolicy();

        CopyKind ck = ep.device == ExecutionPolicy::Device::GPU
                        ? CopyKind::HostToDevice
                        : CopyKind::HostToHost;

        Memcpy(data_, buf, size_ * sizeof(T), ck);

        delete[] buf;
    }

    template <typename T>
    void ManagedBuffer<T>::fill(T const& value)
    {
        fill(const_cast<T&>(value));
    }

    template <typename T>
    void ManagedBuffer<T>::copy(ManagedBuffer& rhs)
    {
        rhs.migrate();

        CopyKind ck = lastAllocationPolicy_.device == ExecutionPolicy::Device::GPU
                        ? CopyKind::DeviceToDevice
                        : CopyKind::HostToHost;

        Memcpy(data_, rhs.data_, std::min(size_, rhs.size_) * sizeof(T), ck);
    }

} // vkt
