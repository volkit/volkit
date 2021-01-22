// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Copy_cuda.hpp"
#include "linalg.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    // Copy kernel that is called when dest and source have different
    // bytes per voxel or different voxel mappings
    __global__ void CopyByValue_kernel(
            StructuredVolumeView dest,
            StructuredVolumeView source,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        int nx = last.x - first.x;
        int ny = last.y - first.y;
        int nz = last.z - first.z;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < nx && y < ny && z < nz)
        {
            dest.setValue(
                x + dstOffset.x,
                y + dstOffset.y,
                z + dstOffset.z,
                source.getValue(x + first.x, y + first.y, z + first.z)
                );
        }
    }

    // Perform bytewise copy when bytes per voxel of dest and source match
    __global__ void CopyBytewise_kernel(
            StructuredVolumeView dest,
            StructuredVolumeView source,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        int nx = last.x - first.x;
        int ny = last.y - first.y;
        int nz = last.z - first.z;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < nx && y < ny && z < nz)
        {
            uint8_t voxel[StructuredVolumeView::GetMaxBytesPerVoxel()];

            source.getBytes(x + first.x, y + first.y, z + first.z, voxel);
            dest.setBytes(
                x + dstOffset.x,
                y + dstOffset.y,
                z + dstOffset.z,
                voxel
                );
        }
    }

    void CopyRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source,
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

        if (dest.getDataFormat() == source.getDataFormat()
          && dest.getVoxelMapping() == source.getVoxelMapping())
        {
            CopyBytewise_kernel<<<gridSize, blockSize>>>(
                    StructuredVolumeView(dest),
                    StructuredVolumeView(source),
                    first,
                    last,
                    dstOffset
                    );
        }
        else
        {
            CopyByValue_kernel<<<gridSize, blockSize>>>(
                    StructuredVolumeView(dest),
                    StructuredVolumeView(source),
                    first,
                    last,
                    dstOffset
                    );
        }
    }

} // vkt
