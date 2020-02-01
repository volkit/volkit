// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <vkt/ManagedBuffer.hpp>

#include "Memory.hpp"

namespace vkt
{
    void ManagedBuffer::migrate()
    {
        ExecutionPolicy ep = GetThreadExecutionPolicy();

        if (ep.device != lastAllocationPolicy_.device)
        {
            uint8_t* newData = nullptr;

            Allocate((void**)&newData, size_);

            CopyKind ck = ep.device == ExecutionPolicy::Device::GPU
                            ? CopyKind::HostToDevice
                            : CopyKind::DeviceToHost;

            Copy(newData, data_, size_, ck);

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

    void ManagedBuffer::resize(std::size_t size)
    {
        std::size_t oldSize = size_;
        std::size_t newSize = size;

        migrate();

        ExecutionPolicy ep = GetThreadExecutionPolicy();

        CopyKind ck = ep.device == ExecutionPolicy::Device::GPU
                        ? CopyKind::DeviceToDevice
                        : CopyKind::HostToHost;

        uint8_t* temp = nullptr;
        Allocate((void**)&temp, oldSize);
        Copy(temp, data_, std::min(oldSize, newSize), ck);

        Free(data_);
        Allocate((void**)data_, newSize);
        Copy(data_, temp, std::min(oldSize, newSize), ck);

        Free(temp);

        size_ = newSize;
    }

    void ManagedBuffer::copy(ManagedBuffer& rhs)
    {
        rhs.migrate();

        CopyKind ck = lastAllocationPolicy_.device == ExecutionPolicy::Device::GPU
                        ? CopyKind::DeviceToDevice
                        : CopyKind::HostToHost;

        Copy(data_, rhs.data_, std::min(size_, rhs.size_), ck);
    }

    void ManagedBuffer::allocate(std::size_t size)
    {
        lastAllocationPolicy_ = GetThreadExecutionPolicy();

        size_ = size;

        Allocate((void**)&data_, size_);
    }

    void ManagedBuffer::deallocate()
    {
        // Free with last allocation policy
        ExecutionPolicy curr = GetThreadExecutionPolicy();

        SetThreadExecutionPolicy(lastAllocationPolicy_);

        Free(data_);

        // Make policy from before Free() call current
        lastAllocationPolicy_ = curr;
        SetThreadExecutionPolicy(curr);
    }

} // vkt
