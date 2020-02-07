// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Arithmetic.hpp>
#include <vkt/StructuredVolume.hpp>

//#include <vkt/Arithmetic.h>
//#include <vkt/StructuredVolume.h>

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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            SumRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SumRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            DiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error DiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            DiffRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            ProdRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error ProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            ProdRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            QuotRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error QuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            QuotRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            AbsDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error AbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            AbsDiffRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            SafeSumRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SafeSumRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SafeSumRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            SafeDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SafeDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SafeDiffRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ)
    {
        VKT_CALL__(
            SafeProdRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SafeProdRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SafeProdRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            SafeQuotRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SafeQuotRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SafeQuotRange,
            dest,
            source1,
            source2,
            first,
            last
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
            dest.getDims()
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
            int32_t lastZ
            )
    {
        VKT_CALL__(
            SafeAbsDiffRange,
            dest,
            source1,
            source2,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error SafeAbsDiffRange(
            StructuredVolume& dest,
            StructuredVolume& source1,
            StructuredVolume& source2,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(
            SafeAbsDiffRange,
            dest,
            source1,
            source2,
            first,
            last
            );

        return NoError;
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

