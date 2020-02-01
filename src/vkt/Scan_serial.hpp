#pragma once

#include <cstring>

#include <vkt/Copy.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include "Voxel.hpp"

namespace vkt
{
    static void ScanRange_serial(
            StructuredVolume& dst,
            StructuredVolume& src,
            vec3i first,
            vec3i last,
            vec3i dstOffset
            )
    {
        static_assert(
                sizeof(uint32_t) >= StructuredVolume::GetMaxBytesPerVoxel(),
                "Type mismatch"
                );

        using accum_t = uint32_t;
        using voxel_t = Voxel<accum_t>;

        auto get = [&](int32_t x, int32_t y, int32_t z)
        {
            uint8_t data[StructuredVolume::GetMaxBytesPerVoxel()];
            dst.getVoxel(x, y, z, data);
            return Voxel<accum_t>(data);
        };

        auto set = [&](int32_t x, int32_t y, int32_t z, accum_t ac)
        {
            uint8_t data[StructuredVolume::GetMaxBytesPerVoxel()];
            std::memcpy(data, &ac, dst.getBytesPerVoxel());
            dst.setVoxel(x, y, z, data);
        };


        // First just copy the whole range from src to dst
        // so we can later only operate on dst
        if (&dst != &src)
        {
            CopyRange(dst, src, first, last, dstOffset);
        }

        // Transform range to "dst space"
        first += dstOffset;
        last  += dstOffset;


        int32_t fx = first.x;
        int32_t fy = first.y;
        int32_t fz = first.z;

        int32_t ix = first.x < last.x ? 1 : -1;
        int32_t iy = first.y < last.y ? 1 : -1;
        int32_t iz = first.z < last.z ? 1 : -1;

        int32_t lx = last.x;
        int32_t ly = last.y;
        int32_t lz = last.z;


        // Init 0-border voxel
        // nop

        // Init 0-border edges (1-d scan)
        for (int32_t x = fx + ix; x != lx; x += ix)
        {
            accum_t ac = get(x, fy, fz) + get(x - ix, fy, fz);

            set(x, fy, fz, ac);
        }

        for (int32_t y = fy + iy; y != ly; y += iy)
        {
            accum_t ac = get(fx, y, fz) + get(fx, y - iy, fz);

            set(fx, y, first.z, ac);
        }

        for (int32_t z = fz + iz; z != lz; z += iz)
        {
            accum_t ac = get(fx, fy, z) + get(fx, fy, z - iz);

            set(fx, fy, z, ac);
        }

        // Init 0-border planes (2-d scan)
        for (int32_t y = fy + iy; y != ly; y += iy)
        {
            for (int32_t x = fx + ix; x != lx; x += ix)
            {
                accum_t ac = get(x, y, fz)
                    + get(x - ix, y, fz) + get(x, y - iy, fz)
                    - get(x - ix, y - iy, fz);

                set(x, y, fz, ac);
            }
        }

        for (int32_t z = fz + iz; z != lz; z += iz)
        {
            for (int32_t y = fy + iy; y != ly; y += iy)
            {
                accum_t ac = get(fx, y, z)
                    + get(fx, y - iy, z) + get(fx, y, z - iz)
                    - get(fx, y - iy, z - iz);

                set(fx, y, z, ac);
            }
        }

        for (int32_t x = fx + ix; x != lx; x += ix)
        {
            for (int32_t z = fz + iz; z != lz; z += iz)
            {
                accum_t ac = get(x, fy, z)
                    + get(x - ix, fy, z) + get(x, fy, z - iz)
                    - get(x - ix, fy, z - iz);

                set(x, fy, z, ac);
            }
        }

        // 3-d scan
        for (int32_t x = first.x + ix; x != last.x; x += ix)
        {
            for (int32_t y = first.y + iy; y != last.y; y += iy)
            {
                for (int32_t z = first.z + iz; z != last.z; z += iz)
                {
                    accum_t ac = get(x, y, z) + get(x - ix, y - iy, z - iz)
                               + get(x - ix, y, z) - get(x, y - iy, z - iz)
                               + get(x, y - iy, z) - get(x - ix, y, z - iz)
                               + get(x, y, z - iz) - get(x - ix, y - iy, z);

                    set(x, y, z, ac);
                }
            }
        }
    }
} // vkt
