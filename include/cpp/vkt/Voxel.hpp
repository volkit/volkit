// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"

namespace vkt
{
    /*!
     * @brief  Voxel view type to pass voxel data around
     */
    struct VoxelView
    {
        uint8_t* bytes;
        uint16_t bytesPerVoxel;
        float mappingLo;
        float mappingHi;
    };

    /*!
     * @brief  provide in-memory representation of mapped voxel
     */
    VKTAPI Error MapVoxel(uint8_t* dst,
                          float value,
                          uint16_t bytesPerVoxel,
                          float mappingLo,
                          float mappingHi);

    /*!
     * @brief  convert from mapped in-memory representation to voxel value
     */
    VKTAPI Error UnmapVoxel(float& value,
                            uint8_t const* src,
                            uint16_t bytesPerVoxel,
                            float mappingLo,
                            float mappingHi);

} // vkt
