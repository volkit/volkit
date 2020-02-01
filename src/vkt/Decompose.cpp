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
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& source,
            int brickSizeX,
            int brickSizeY,
            int brickSizeZ,
            int haloSizeNegX,
            int haloSizeNegY,
            int haloSizeNegZ,
            int haloSizePosX,
            int haloSizePosY,
            int haloSizePosZ
            )
    {
        VKT_CALL__(
                BrickDecompose,
                decomp,
                source,
                Vec3i(brickSizeX, brickSizeY, brickSizeZ),
                Vec3i(haloSizeNegX, haloSizeNegY, haloSizeNegZ),
                Vec3i(haloSizePosX, haloSizePosY, haloSizePosZ)
                );

        return NO_ERROR;
    }

    Error BrickDecompose(
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& source,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        VKT_CALL__(BrickDecompose, decomp, source, brickSize, haloSizeNeg, haloSizePos);

        return NO_ERROR;
    }

    Error BrickDecomposeGetNumBricks(
            int32_t& numBricksX,
            int32_t& numBricksY,
            int32_t& numBricksZ,
            int32_t dimX,
            int32_t dimY,
            int32_t dimZ,
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
        (void)haloSizeNegX;
        (void)haloSizeNegY;
        (void)haloSizeNegZ;
        (void)haloSizePosX;
        (void)haloSizePosY;
        (void)haloSizePosZ;

        numBricksX = div_up(dimX, brickSizeX);
        numBricksY = div_up(dimY, brickSizeY);
        numBricksZ = div_up(dimZ, brickSizeZ);

        return NO_ERROR;
    }

    Error BrickDecomposeGetNumBricks(
            Vec3i& numBricks,
            Vec3i dims,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
        BrickDecomposeGetNumBricks(
                numBricks.x,
                numBricks.y,
                numBricks.z,
                dims.x,
                dims.y,
                dims.z,
                brickSize.x,
                brickSize.y,
                brickSize.z,
                haloSizeNeg.x,
                haloSizeNeg.y,
                haloSizeNeg.z,
                haloSizePos.x,
                haloSizePos.y,
                haloSizePos.z
                );

        return NO_ERROR;
    }

} // vkt
