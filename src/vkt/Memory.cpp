#include "macros.hpp"
#include "Memory.hpp"
#include "Memory_cuda.hpp"
#include "Memory_serial.hpp"

namespace vkt
{
    void Allocate(void** ptr, std::size_t size)
    {
        VKT_CALL__(Allocate, ptr, size);
    }

    void Free(void* ptr)
    {
        VKT_CALL__(Free, ptr);
    }

    void Copy(void* dst, void const* src, std::size_t size, CopyKind ck)
    {
        VKT_CALL__(Copy, dst, src, size, ck);
    }

} // vkt
