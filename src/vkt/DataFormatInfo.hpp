// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/common.hpp>

#include "macros.hpp"

namespace vkt
{
    struct DataFormatInfo
    {
        DataFormat dataFormat;
        uint8_t sizeInBytes;
    };

    static DataFormatInfo DataFormatInfoTable[(int)DataFormat::Count] = {
            { DataFormat::Unspecified,  0 },
            { DataFormat::Int8,         1 },
            { DataFormat::Int16,        2 },
            { DataFormat::Int32,        4 },
            { DataFormat::UInt8,        1 },
            { DataFormat::UInt16,       2 },
            { DataFormat::UInt32,       4 },
            { DataFormat::Float32,      4 },

    };

    // Equivalent to table, but can be used in CUDA device code
    VKT_FUNC constexpr inline uint8_t getSizeInBytes(DataFormat dataFormat)
    {
       if (dataFormat == DataFormat::Int8 || dataFormat == DataFormat::UInt8)
           return 1;

       if (dataFormat == DataFormat::Int16 || dataFormat == DataFormat::UInt16)
           return 2;

       if (dataFormat == DataFormat::Int32 || dataFormat == DataFormat::UInt32)
           return 4;

       if (dataFormat == DataFormat::Float32)
           return 4;

       return 255;
    }
} // vkt
