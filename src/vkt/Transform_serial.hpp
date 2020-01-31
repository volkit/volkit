#pragma once

#include <cstring> // memset

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    // Unary op
    inline void TransformRange_serial(
            StructuredVolume& volume,
            vec3i first,
            vec3i last,
            TransformUnaryOp unaryOp
            )
    {
        uint8_t voxel[StructuredVolume::GetMaxBytesPerVoxel()];

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    std::memset(
                        &voxel,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    volume.getVoxel(x, y, z, voxel);
                    unaryOp(x, y, z, voxel);
                    volume.setVoxel(x, y, z, voxel);
                }
            }
        }
    }

    // Binary op
    inline void TransformRange_serial(
            StructuredVolume& volume1,
            StructuredVolume& volume2,
            vec3i first,
            vec3i last,
            TransformBinaryOp binaryOp
            )
    {
        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    uint8_t voxel1[StructuredVolume::GetMaxBytesPerVoxel()];
                    std::memset(
                        &voxel1,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    uint8_t voxel2[StructuredVolume::GetMaxBytesPerVoxel()];
                    std::memset(
                        &voxel2,
                        0,
                        sizeof(uint8_t) * StructuredVolume::GetMaxBytesPerVoxel()
                        );

                    volume1.getVoxel(x, y, z, voxel1);
                    volume2.getVoxel(x, y, z, voxel2);
                    binaryOp(x, y, z, voxel1, voxel2);
                    volume1.setVoxel(x, y, z, voxel1);
                    volume2.setVoxel(x, y, z, voxel2);
                }
            }
        }
    }
} // vkt
