// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Decompose.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Decompose.h>
#include <vkt/StructuredVolume.h>

#include "Decompose_cuda.hpp"
#include "Decompose_serial.hpp"
#include "macros.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error BrickDecompose(
            Array3D<StructuredVolume>& dest,
            StructuredVolume& source,
            int32_t brickSizeX,
            int32_t brickSizeY,
            int32_t brickSizeZ,
            int32_t haloSizeNegX,
            int32_t haloSizeNegY,
            int32_t haloSizeNegZ,
            int32_t haloSizePosX,
            int32_t haloSizePosY,
            int32_t haloSizePosZ
            )
    {
        VKT_CALL__(
                BrickDecompose,
                dest,
                source,
                Vec3i(brickSizeX, brickSizeY, brickSizeZ),
                Vec3i(haloSizeNegX, haloSizeNegY, haloSizeNegZ),
                Vec3i(haloSizePosX, haloSizePosY, haloSizePosZ)
                );

        return NoError;
    }

    Error BrickDecompose(
            Array3D<StructuredVolume>& dest,
            StructuredVolume& source,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        VKT_CALL__(BrickDecompose, dest, source, brickSize, haloSizeNeg, haloSizePos);

        return NoError;
    }

    Error BrickDecomposeResize(
            Array3D<StructuredVolume>& dest,
            StructuredVolume& source,
            int32_t brickSizeX,
            int32_t brickSizeY,
            int32_t brickSizeZ,
            int32_t haloSizeNegX,
            int32_t haloSizeNegY,
            int32_t haloSizeNegZ,
            int32_t haloSizePosX,
            int32_t haloSizePosY,
            int32_t haloSizePosZ
            )
    {
        return BrickDecomposeResize(
            dest,
            source,
            Vec3i(brickSizeX, brickSizeY, brickSizeZ),
            Vec3i(haloSizeNegX, haloSizeNegY, haloSizeNegZ),
            Vec3i(haloSizePosX, haloSizePosY, haloSizePosZ)
            );
    }

    Error BrickDecomposeResize(
            Array3D<StructuredVolume>& dest,
            StructuredVolume& source,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        Vec3i sourceDims = source.getDims();

        Vec3i numBricks(
            div_up(sourceDims.x, brickSize.x),
            div_up(sourceDims.y, brickSize.y),
            div_up(sourceDims.z, brickSize.z)
            );

        // That's the size a minimal extended volume would have
        // that could accommodate numBricks
        Vec3i extendedDims = numBricks * brickSize;

        // That's, accordingly, the brick size (w/o halos) at the rightmost borders
        Vec3i borderSize(
            sourceDims.x % brickSize.x == 0 ? brickSize.x : brickSize.x - extendedDims.x + sourceDims.x,
            sourceDims.y % brickSize.y == 0 ? brickSize.y : brickSize.y - extendedDims.y + sourceDims.y,
            sourceDims.z % brickSize.z == 0 ? brickSize.z : brickSize.z - extendedDims.z + sourceDims.z
            );

        // Allocate storage for numBricks bricks with halos
        dest = vkt::Array3D<vkt::StructuredVolume>(numBricks);
        for (int z = 0; z < numBricks.z; ++z)
        {
            for (int y = 0; y < numBricks.y; ++y)
            {
                for (int x = 0; x < numBricks.x; ++x)
                {
                    Vec3i index(x, y, z);

                    // Crop brick size according to volume boundaries

                    Vec3i size(
                        x < numBricks.x - 1 ? brickSize.x : borderSize.x,
                        y < numBricks.y - 1 ? brickSize.y : borderSize.y,
                        z < numBricks.z - 1 ? brickSize.z : borderSize.z
                        );

                    dest[index] = vkt::StructuredVolume(
                        haloSizeNeg.x + size.x + haloSizePos.x,
                        haloSizeNeg.y + size.y + haloSizePos.y,
                        haloSizeNeg.z + size.z + haloSizePos.z,
                        source.getBytesPerVoxel(),
                        source.getDist().x,
                        source.getDist().y,
                        source.getDist().z,
                        source.getVoxelMapping().x,
                        source.getVoxelMapping().y
                        );
                }
            }
        }

        return NoError;
    }

} // vkt
