// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Arithmetic.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Arithmetic.h>
#include <vkt/StructuredVolume.h>

#include "Arithmetic_cuda.hpp"
#include "Arithmetic_serial.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Sum(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SumRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SumRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SumRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error Diff(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            DiffRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error DiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            DiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error DiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            DiffRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error Prod(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            ProdRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error ProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            ProdRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error ProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            ProdRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error Quot(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            QuotRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error QuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            QuotRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error QuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            QuotRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error AbsDiff(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            AbsDiffRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error AbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            AbsDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error AbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            AbsDiffRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error SafeSum(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SafeSumRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SafeSumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SafeSumRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SafeSumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SafeSumRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error SafeDiff(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SafeDiffRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SafeDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SafeDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SafeDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SafeDiffRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error SafeProd(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SafeProdRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SafeProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SafeProdRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SafeProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SafeProdRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error SafeQuot(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SafeQuotRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SafeQuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SafeQuotRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SafeQuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SafeQuotRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

    Error SafeAbsDiff(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2
            )
    {
        VKT_CALL__(
            SafeAbsDiffRange,
            dest,
            source1,
            source2,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error SafeAbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
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
            SafeAbsDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error SafeAbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(
            SafeAbsDiffRange,
            dest,
            source1,
            source2,
            first,
            last,
            dstOffset
            );

        return NoError;
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktSumSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::Sum(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSumRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SumRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktDiffSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::Diff(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktDiffRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::DiffRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktProdSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::Prod(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktProdRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::ProdRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktQuotSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::Quot(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktQuotRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::QuotRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktAbsDiffSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::AbsDiff(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktAbsDiffRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::AbsDiffRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktSafeSumSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::SafeSum(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSafeSumRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SafeSumRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktSafeDiffSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::SafeDiff(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSafeDiffRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SafeDiffRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktSafeProdSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::SafeProd(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSafeProdRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SafeProdRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktSafeQuotSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::SafeQuot(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSafeQuotRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SafeQuotRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}

vktError vktSafeAbsDiffSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2
        )
{
    return (vktError)vkt::SafeAbsDiff(
            dest->volume,
            source1->volume,
            source2->volume
            );
}

vktError vktSafeAbsDiffRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source1,
        vktStructuredVolume source2,
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
    return (vktError)vkt::SafeAbsDiffRange(
            dest->volume,
            source1->volume,
            source2->volume,
            firstX,
            firstY,
            firstZ,
            lastX,
            lastY,
            lastZ,
            dstOffsetX,
            dstOffsetY,
            dstOffsetZ
            );
}
