// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <vkt/Histogram.hpp>
#include <vkt/StructuredVolume.hpp>

#include "Callable.hpp"
#include "Histogram_serial.hpp"
#include "StructuredVolume_impl.hpp"

#if VKT_HAVE_CUDA
#include "Histogram_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Histogram::Histogram(std::size_t numBins)
    {
        resize(numBins);
    }

    std::size_t Histogram::getNumBins() const
    {
        return size_;
    }

    std::size_t* Histogram::getBinCounts()
    {
        migrate();

        return data_;
    }

    Error ComputeHistogram(StructuredVolume& volume, Histogram& histogram)
    {
        VKT_LEGACY_CALL__(ComputeHistogramRange, volume, histogram, { 0, 0, 0 }, volume.getDims());

        return NoError;
    }

    Error ComputeHistogramRange(
            StructuredVolume& volume,
            Histogram& histogram,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ
            )
    {
        VKT_LEGACY_CALL__(
            ComputeHistogramRange,
            volume,
            histogram,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error ComputeHistogramRange(
            StructuredVolume& volume,
            Histogram& histogram,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_LEGACY_CALL__(
            ComputeHistogramRange,
            volume,
            histogram,
            first,
            last
            );

        return NoError;
    }
} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

