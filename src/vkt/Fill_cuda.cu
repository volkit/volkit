#include <cstddef>
#include <cstdint>

#include "Fill_cuda.hpp"
#include "macros.hpp"

namespace vkt
{
    __constant__ uint8_t deviceMappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];

    __global__ void Fill_kernel(
            uint8_t* data,
            vec3i dims,
            uint16_t bytesPerVoxel,
            vec3i first,
            vec3i last
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
            std::size_t linearIndex = z * static_cast<std::size_t>(dims.x) * dims.y
                                    + y * dims.x
                                    + x;
            linearIndex *= bytesPerVoxel;

            for (uint16_t i = 0; i < bytesPerVoxel; ++i)
                data[linearIndex + i] = deviceMappedVoxel[i];
        }
    }

    void FillRange_cuda(StructuredVolume& volume, vec3i first, vec3i last, float value)
    {
        uint8_t mappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];
        volume.mapVoxel(mappedVoxel, value);

        VKT_CUDA_SAFE_CALL__(cudaMemcpyToSymbol(
                deviceMappedVoxel,
                mappedVoxel,
                StructuredVolume::GetMaxBytesPerVoxel(),
                0,
                cudaMemcpyHostToDevice
                ));

        unsigned nx = last.x - first.x;
        unsigned ny = last.y - first.y;
        unsigned nz = last.z - first.z;

        dim3 blockSize(8, 8, 8);
        dim3 gridSize(
                div_up(nx, blockSize.x),
                div_up(ny, blockSize.y),
                div_up(nz, blockSize.z)
                );

        Fill_kernel<<<gridSize, blockSize>>>(
                volume.getData(),
                volume.getDims(),
                volume.getBytesPerVoxel(),
                first,
                last
                );
    }
} // vkt
