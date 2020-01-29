#pragma once

#include <cstddef>

namespace vkt
{
    enum class CopyKind
    {
        HostToHost,
        HostToDevice,
        DeviceToHost,
        DeviceToDevice,
    };

    void Allocate(void** ptr, std::size_t size);

    void Free(void* ptr);

    void Copy(void* dst, void const* src, std::size_t size, CopyKind ck);

} // vkt
