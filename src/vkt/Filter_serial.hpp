// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include "FilterView.hpp"
#include "linalg.hpp"

namespace vkt
{
    void ApplyFilterRange_serial(
        StructuredVolume& dest,
        StructuredVolume& source,
        Vec3i first,
        Vec3i last,
        Filter filter,
        AddressMode am
        )
    {
        FilterView f(filter);

        // TODO: better just copy blocks
        // TODO: don't need temp volume if source != dest!!
        StructuredVolume temp(source);

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    // TODO: mind address mode!
                    // TODO: this assumes that filter size is 3x3x3....
                    if (x == 0 || y == 0 || z == 0
                        || x == last.x-1 || y == last.y-1 || z == last.z-1)
                    {
                        dest.setValue(x, y, z, 0.f);
                    }
                    else
                    {
                        float value = 0.f;
                        Vec3i filterDims2 = filter.getDims()/Vec3i{2,2,2};
                        for (int32_t zz = 0; zz != filter.getDims().z; ++zz)
                        {
                            for (int32_t yy = 0; yy != filter.getDims().y; ++yy)
                            {
                                for (int32_t xx = 0; xx != filter.getDims().x; ++xx)
                                {
                                    value += f(xx, yy, zz)
                                        * temp.getValue(x-filterDims2.x+xx,
                                                        y-filterDims2.y+yy,
                                                        z-filterDims2.z+zz);
                                }
                            }
                        }
                        dest.setValue(x, y, z, value);
                    }
                }
            }
        }
    }

} // vkt
