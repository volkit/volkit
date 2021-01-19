// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void SumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void DiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void ProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void QuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void AbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void SafeSumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void SafeDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void SafeProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void SafeQuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void SafeAbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

} // vkt
