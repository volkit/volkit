// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstdint>

#include <vkt/Voxel.hpp>

#include "DataFormatInfo.hpp"
#include "Fill_cuda.hpp"
#include "for_each.hpp"
#include "linalg.hpp"
#include "macros.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    using serial::for_each;

    __constant__ uint8_t deviceMappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];

    void FillRange_cuda(StructuredVolume& volume, Vec3i first, Vec3i last, float value)
    {
        uint8_t mappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];
        MapVoxel(
            mappedVoxel,
            value,
            volume.getDataFormat(),
            volume.getVoxelMapping().x,
            volume.getVoxelMapping().y
            );

        VKT_CUDA_SAFE_CALL__(cudaMemcpyToSymbol(
                deviceMappedVoxel,
                mappedVoxel,
                StructuredVolume::GetMaxBytesPerVoxel(),
                0,
                cudaMemcpyHostToDevice
                ));

        StructuredVolumeView view(volume);
        for_each(first.x,last.x,first.y,last.y,first.z,last.z,
                 [=] __device__ (int x, int y, int z) mutable {
                     std::size_t bytesPerVoxel = getSizeInBytes(view.getDataFormat());
                     Vec3i dims = view.getDims();
                     uint8_t* data = (uint8_t*)view.getData();

                     std::size_t linearIndex = z * static_cast<std::size_t>(dims.x) * dims.y
                                             + y * dims.x
                                             + x;
                     linearIndex *= bytesPerVoxel;

                     for (std::size_t i = 0; i < bytesPerVoxel; ++i)
                         data[linearIndex + i] = deviceMappedVoxel[i];
                });
    }

    void FillRange_cuda(HierarchicalVolume& volume, Vec3i first, Vec3i last, float value)
    {
    }
} // vkt
