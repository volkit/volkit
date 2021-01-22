// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstring> // memset

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/Voxel.hpp>

namespace vkt
{
    // Unary op
    inline void TransformRange_serial(
            StructuredVolume& volume,
            Vec3i first,
            Vec3i last,
            TransformUnaryOp unaryOp
            )
    {
        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    uint8_t bytes[StructuredVolume::GetMaxBytesPerVoxel()];

                    std::memset(
                        &bytes,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    volume.getBytes(x, y, z, bytes);

                    VoxelView voxel;
                    voxel.bytes = bytes;
                    voxel.dataFormat = volume.getDataFormat();
                    voxel.mappingLo = volume.getVoxelMapping().x;
                    voxel.mappingHi = volume.getVoxelMapping().y;

                    unaryOp(x, y, z, voxel);

                    volume.setBytes(x, y, z, bytes);
                }
            }
        }
    }

    // Binary op
    inline void TransformRange_serial(
            StructuredVolume& volume1,
            StructuredVolume& volume2,
            Vec3i first,
            Vec3i last,
            TransformBinaryOp binaryOp
            )
    {
        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    uint8_t bytes1[StructuredVolume::GetMaxBytesPerVoxel()];
                    std::memset(
                        &bytes1,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    uint8_t bytes2[StructuredVolume::GetMaxBytesPerVoxel()];
                    std::memset(
                        &bytes2,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    volume1.getBytes(x, y, z, bytes1);
                    volume2.getBytes(x, y, z, bytes2);

                    VoxelView voxel1;
                    voxel1.bytes = bytes1;
                    voxel1.dataFormat = volume1.getDataFormat();
                    voxel1.mappingLo = volume1.getVoxelMapping().x;
                    voxel1.mappingHi = volume1.getVoxelMapping().y;

                    VoxelView voxel2;
                    voxel2.bytes = bytes2;
                    voxel2.dataFormat = volume2.getDataFormat();
                    voxel2.mappingLo = volume2.getVoxelMapping().x;
                    voxel2.mappingHi = volume2.getVoxelMapping().y;

                    binaryOp(x, y, z, voxel1, voxel2);

                    volume1.setBytes(x, y, z, bytes1);
                    volume2.setBytes(x, y, z, bytes2);
                }
            }
        }
    }
} // vkt
