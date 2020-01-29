#pragma once

#include <cstdint>

#include <vkt/StructuredVolume.hpp>

struct vktStructuredVolume_impl
{
    vktStructuredVolume_impl(
            int32_t dimx,
            int32_t dimy,
            int32_t dimz,
            uint16_t bytesPerVoxel,
            float mappingLo,
            float mappingHi)
        : volume(dimx, dimy, dimz, bytesPerVoxel, mappingLo, mappingHi)
    {
    }

    vktStructuredVolume_impl(vkt::StructuredVolume& rhs)
        : volume(rhs)
    {
    }

    vkt::StructuredVolume volume;
};
