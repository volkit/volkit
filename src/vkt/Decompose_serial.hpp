// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Array3D.hpp>
#include <vkt/Copy.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    static void BrickDecompose_serial(
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& volume,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        for (int z = 0; z < decomp.dims().z; ++z)
        {
            for (int y = 0; y < decomp.dims().y; ++y)
            {
                for (int x = 0; x < decomp.dims().x; ++x)
                {
                    Vec3i index(x, y, z);
                    Vec3i first = index * brickSize;
                    Vec3i last = first + brickSize;
                    first -= haloSizeNeg;
                    last += haloSizePos;

                    CopyRange(
                        decomp[index],
                        volume,
                        first,
                        last
                        );
                }
            }
        }
    }
} // vkt
