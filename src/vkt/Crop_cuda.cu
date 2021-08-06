// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Crop_cuda.hpp"

#include "DataFormatInfo.hpp"
#include "HierarchicalVolumeView.hpp"
#include "linalg.hpp"

namespace vkt
{
    __global__ void Crop_kernel(
            HierarchicalVolumeView dst,
            HierarchicalVolumeView src,
            Vec3i first,
            Vec3i last,
            int* brickIDs
            )
    {
        std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < dst.getNumBricks())
        {
            Brick oldBrick = src.getBricks()[brickIDs[i]];
            Brick newBrick = dst.getBricks()[i];

            unsigned levelDiff = oldBrick.level-newBrick.level;
            // That's by how much we have to multiply the
            // new cell size to obtain the old cell size
            unsigned sizeDiff = 1<<levelDiff;

            Vec3i loDiff = (newBrick.lower+first)-oldBrick.lower;

            std::size_t offNew = newBrick.offsetInBytes;

            for (int z = 0; z < newBrick.dims.z; z += sizeDiff)
            {
                for (int y = 0; y < newBrick.dims.y; y += sizeDiff)
                {
                    for (int x = 0; x < newBrick.dims.x; x += sizeDiff)
                    {
                        int oldX = (loDiff.x>>oldBrick.level) + x / sizeDiff;
                        int oldY = (loDiff.y>>oldBrick.level) + y / sizeDiff;
                        int oldZ = (loDiff.z>>oldBrick.level) + z / sizeDiff;

                        std::size_t off = oldBrick.offsetInBytes
                            + (oldZ * oldBrick.dims.x * oldBrick.dims.y
                             + oldY * oldBrick.dims.x
                             + oldX) * vkt::getSizeInBytes(src.getDataFormat());

                        uint8_t bytes[HierarchicalVolume::GetMaxBytesPerVoxel()];

                        memcpy(bytes, src.getData() + off,
                               vkt::getSizeInBytes(src.getDataFormat()));

                        for (unsigned d = 0; d < sizeDiff; ++d)
                        {
                            std::memcpy((void*)dst.getData() + offNew, bytes,
                                        vkt::getSizeInBytes(src.getDataFormat()));
                            offNew += vkt::getSizeInBytes(src.getDataFormat());
                        }
                    }
                }
            }
        }
    }

    void Crop_cuda(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            Vec3i first,
            Vec3i last
            )
    {
        dim3 blockSize(1024);
        dim3 gridSize(div_up((unsigned)dst.getNumBricks(), blockSize.x));

        // Retrieve the src brickIDs that were temporarily stored
        // in the scalar field
        int* brickIDs = nullptr;
        cudaMalloc(&brickIDs, sizeof(int) * dst.getNumBricks());
        cudaMemcpy(brickIDs, dst.getData(), sizeof(int) * dst.getNumBricks(),
                   cudaMemcpyDeviceToDevice);

        // Default construct those, they're unused anyway
        HierarchicalVolumeAccel dstAccel;
        HierarchicalVolumeAccel srcAccel;

        Crop_kernel<<<gridSize, blockSize>>>(
                HierarchicalVolumeView(dst, dstAccel),
                HierarchicalVolumeView(src, srcAccel),
                first,
                last,
                brickIDs
                );

        cudaFree(brickIDs);
    }
} // vkt
