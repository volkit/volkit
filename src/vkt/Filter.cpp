// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cstring>

#include <vkt/Filter.hpp>
#include <vkt/Memory.hpp>
#include <vkt/StructuredVolume.hpp>

#include "Filter_serial.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

#if VKT_HAVE_CUDA
#include "Filter_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Filter::Filter(float* data, Vec3i dims)
        : ManagedBuffer(dims.x * size_t(dims.y) * dims.z * sizeof(float))
        , dims_(dims)
    {
        // TODO: this won't work if current execution policy is GPU
        memcpy(getData(), data, dims.x * size_t(dims.y) * dims.z * sizeof(float));

        migrate();
    }

    float* Filter::getData()
    {
        return (float*)data_;
    }

    Vec3i Filter::getDims() const
    {
        return dims_;
    }

    Error ApplyFilter(
        StructuredVolume& dest,
        StructuredVolume& source,
        Filter& filter,
        AddressMode am
        )
    {
        VKT_CALL__(
            ApplyFilterRange,
            dest,
            source,
            { 0, 0, 0 },
            source.getDims(),
            filter,
            am
            );

        return NoError;
    }

} // vkt
