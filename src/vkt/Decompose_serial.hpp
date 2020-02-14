// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Array3D.hpp>
#include <vkt/Copy.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"
#include "StructuredVolume_impl.hpp"

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

    // TODO: deduplicate!!
    static void BrickDecomposeC_serial(
            vktArray3D_vktStructuredVolume dest,
            vktStructuredVolume source,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        vktVec3i_t dimsC = vktArray3D_vktStructuredVolume_Dims(dest);
        Vec3i dims{dimsC.x, dimsC.y, dimsC.z};

        for (int z = 0; z < dims.z; ++z)
        {
            for (int y = 0; y < dims.y; ++y)
            {
                for (int x = 0; x < dims.x; ++x)
                {
                    Vec3i index{x, y, z};
                    Vec3i first = index * brickSize;
                    Vec3i last{
                        std::min(first.x + brickSize.x, source->volume.getDims().x),
                        std::min(first.y + brickSize.y, source->volume.getDims().y),
                        std::min(first.z + brickSize.z, source->volume.getDims().z)
                        };
                    first -= haloSizeNeg;
                    last += haloSizePos;

                    vktVec3i_t indexC{x, y, z};
                    CopyRange(
                        (*vktArray3D_vktStructuredVolume_Access(dest, indexC))->volume,
                        source->volume,
                        first,
                        last
                        );
                }
            }
        }
    }
} // vkt
