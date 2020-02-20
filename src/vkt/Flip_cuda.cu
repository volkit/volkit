// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>

#include "Flip_cuda.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    __global__ void Flip_kernel(
            StructuredVolumeView dest,
            StructuredVolumeView source,
            Axis axis,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        int nx = last.x - first.x;
        int ny = last.y - first.y;
        int nz = last.z - first.z;

        int x = (blockIdx.x * blockDim.x + threadIdx.x) - first.x;
        int y = (blockIdx.y * blockDim.y + threadIdx.y) - first.y;
        int z = (blockIdx.z * blockDim.z + threadIdx.z) - first.z;

        if (x < nx && y < ny && z < nz)
        {
            int xx = (axis == Axis::X) ? last.x - 1 - x : x;
            int yy = (axis == Axis::Y) ? last.y - 1 - y : y;
            int zz = (axis == Axis::Y) ? last.z - 1 - z : z;

            uint8_t voxel1[StructuredVolumeView::GetMaxBytesPerVoxel()];
            uint8_t voxel2[StructuredVolumeView::GetMaxBytesPerVoxel()];

            // Exchange in a way that would even work if dest eq source
            source.getBytes(x, y, z, voxel1);
            source.getBytes(xx, yy, zz, voxel2);

            dest.setBytes(dstOffset.x + x, dstOffset.y + y, dstOffset.z + z, voxel2);
            dest.setBytes(dstOffset.x + xx, dstOffset.y + yy, dstOffset.z + zz, voxel1);
        }
    }

    void FlipRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source,
            Axis axis,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        unsigned nx = last.x - first.x;
        unsigned ny = last.y - first.y;
        unsigned nz = last.z - first.z;

        dim3 blockSize(8, 8, 8);
        dim3 gridSize(
                div_up(nx, blockSize.x),
                div_up(ny, blockSize.y),
                div_up(nz, blockSize.z)
                );

        Flip_kernel<<<gridSize, blockSize>>>(
                StructuredVolumeView(dest),
                StructuredVolumeView(source),
                axis,
                first,
                last,
                dstOffset
                );
    }

} // vkt
