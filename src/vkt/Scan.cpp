// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Scan.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Scan.h>
#include <vkt/StructuredVolume.h>

#include "macros.hpp"
#include "Scan_cuda.hpp"
#include "Scan_serial.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Scan(StructuredVolume& dst, StructuredVolume& src)
    {
        VKT_CALL__(
            ScanRange,
            dst,
            src,
            Vec3i(0, 0, 0),
            dst.getDims(),
            Vec3i(0, 0, 0)
            );

        return NO_ERROR;
    }

    Error ScanRange(
            StructuredVolume& dst,
            StructuredVolume& src,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ,
            int32_t dstOffsetX,
            int32_t dstOffsetY,
            int32_t dstOffsetZ
            )
    {
        VKT_CALL__(
            ScanRange,
            dst,
            src,
            Vec3i(firstX, firstY, firstZ),
            Vec3i(lastX, lastY, lastZ),
            Vec3i(dstOffsetX, dstOffsetY, dstOffsetZ)
            );

        return NO_ERROR;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktScanSV(vktStructuredVolume dst, vktStructuredVolume src)
{
    VKT_CALL__(
        ScanRange,
        dst->volume,
        src->volume,
        vkt::Vec3i(0, 0, 0),
        dst->volume.getDims(),
        vkt::Vec3i(0, 0, 0)
        );

    return VKT_NO_ERROR;
}

vktError vktScanRangeSV(
        vktStructuredVolume dst,
        vktStructuredVolume src,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        int32_t dstOffsetX,
        int32_t dstOffsetY,
        int32_t dstOffsetZ
        )
{
    VKT_CALL__(
        ScanRange,
        dst->volume,
        src->volume,
        vkt::Vec3i(firstX, firstY, firstZ),
        vkt::Vec3i(lastX, lastY, lastZ),
        vkt::Vec3i(dstOffsetX, dstOffsetY, dstOffsetZ)
        );

    return VKT_NO_ERROR;
}
