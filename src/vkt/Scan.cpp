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
            vec3i(0, 0, 0),
            dst.getDims(),
            vec3i(0, 0, 0)
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
            vec3i(firstX, firstY, firstZ),
            vec3i(lastX, lastY, lastZ),
            vec3i(dstOffsetX, dstOffsetY, dstOffsetZ)
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
        vkt::vec3i(0, 0, 0),
        dst->volume.getDims(),
        vkt::vec3i(0, 0, 0)
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
        vkt::vec3i(firstX, firstY, firstZ),
        vkt::vec3i(lastX, lastY, lastZ),
        vkt::vec3i(dstOffsetX, dstOffsetY, dstOffsetZ)
        );

    return VKT_NO_ERROR;
}
