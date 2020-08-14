// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Scale.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"

namespace vkt
{
    void ScaleRange_serial(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Vec3f scalingFactor,
            Vec3f centerOfScaling
            )
    {
        Mat3f s = {
            { 1.f / scalingFactor.x, 0.f, 0.f },
            { 0.f, 1.f / scalingFactor.y, 0.f },
            { 0.f, 0.f, 1.f / scalingFactor.z }
            };

        // Iterate over the _whole_ dest volume,
        // apply the _inverse_ scaling, and reconstruct
        // if the scaled position is inside [first..last)
        for (int z = 0; z < dest.getDims().z; ++z)
        {
            for (int y = 0; y < dest.getDims().y; ++y)
            {
                for (int x = 0; x < dest.getDims().x; ++x)
                {
                    Vec3f p{ (float)x, (float)y, float(z) };
                    p = p - centerOfScaling;
                    p = s * p;
                    p = p + centerOfScaling;

                    if (p.x >= first.x && p.x < last.x
                     && p.y >= first.y && p.y < last.y
                     && p.z >= first.z && p.z < last.z)
                    {
                        float val = source.sampleLinear(p.x, p.y, p.z);
                        dest.setValue(x, y, z, val);
                    }
                }
            }
        }
    }
} // vkt
