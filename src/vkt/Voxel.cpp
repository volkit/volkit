// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/common.hpp>
#include <vkt/linalg.hpp>
#include <vkt/Voxel.hpp>

#include <vkt/System.h>
#include <vkt/Voxel.h>

#include "VoxelMapping.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error MapVoxel(
            uint8_t* dst,
            float value,
            DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
    {
        MapVoxelImpl(dst, value, dataFormat, mappingLo, mappingHi);

        return NoError;
    }

    Error UnmapVoxel(
            float& value,
            uint8_t const* src,
            DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
    {
        UnmapVoxelImpl(value, src, dataFormat, mappingLo, mappingHi);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktMapVoxel(
        uint8_t* dst,
        float value,
        vktDataFormat dataFormat,
        float mappingLo,
        float mappingHi
        )
{
    return (vktError)vkt::MapVoxel(dst, value, (vkt::DataFormat)dataFormat, mappingLo, mappingHi);
}

vktError vktUnmapVoxel(
        float* value,
        uint8_t const* src,
        vktDataFormat dataFormat,
        float mappingLo,
        float mappingHi
        )
{
    return (vktError)vkt::UnmapVoxel(*value, src, (vkt::DataFormat)dataFormat, mappingLo, mappingHi);
}
