// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Array3D.hpp>
#include <vkt/Copy.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"

namespace vkt
{
    static void BrickDecompose_serial(
            Array3D<StructuredVolume>& dest,
            StructuredVolume& source,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        for (int z = 0; z < dest.dims().z; ++z)
        {
            for (int y = 0; y < dest.dims().y; ++y)
            {
                for (int x = 0; x < dest.dims().x; ++x)
                {
                    Vec3i index{x, y, z};
                    Vec3i first = index * brickSize;
                    Vec3i last{
                        std::min(first.x + brickSize.x, source.getDims().x),
                        std::min(first.y + brickSize.y, source.getDims().y),
                        std::min(first.z + brickSize.z, source.getDims().z)
                        };
                    first -= haloSizeNeg;
                    last += haloSizePos;

                    CopyRange(
                        dest[index],
                        source,
                        first,
                        last
                        );
                }
            }
        }
    }
} // vkt
