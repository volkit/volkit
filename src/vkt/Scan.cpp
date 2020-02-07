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
            { 0, 0, 0 },
            dst.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
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
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
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
        { 0, 0, 0 },
        dst->volume.getDims(),
        { 0, 0, 0 }
        );

    return vktNoError;
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
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        { dstOffsetX, dstOffsetY, dstOffsetZ }
        );

    return vktNoError;
}
