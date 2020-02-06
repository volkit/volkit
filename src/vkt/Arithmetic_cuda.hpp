// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void SumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void DiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void ProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void QuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void AbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void SafeSumRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void SafeDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void SafeProdRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void SafeQuotRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

    void SafeAbsDiffRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            );

} // vkt
