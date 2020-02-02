// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/LookupTable.hpp>

struct vktLookupTable_impl
{
    vktLookupTable_impl(
            int32_t dimx,
            int32_t dimy,
            int32_t dimz,
            vkt::ColorFormat format
            )
        : lut(dimx, dimy, dimz, format)
    {
    }

    vktLookupTable_impl(vkt::LookupTable& rhs)
        : lut(rhs)
    {
    }

    vkt::LookupTable lut;
};
