// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/common.hpp>
#include <vkt/Histogram.hpp>

#include "macros.hpp"

namespace vkt
{
    class HistogramView
    {
    public:
        HistogramView() = default;

        VKT_FUNC HistogramView(Histogram& histogram)
            : numBins_(histogram.getNumBins())
            , binCounts_(histogram.getBinCounts())
        {
        }

        VKT_FUNC std::size_t getNumBins() const
        {
            return numBins_;
        }

        VKT_FUNC std::size_t* getBinCounts()
        {
            return binCounts_;
        }

    private:
        std::size_t numBins_;

        std::size_t* binCounts_;
    };
} // vkt
