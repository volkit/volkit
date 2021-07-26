// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>

#include <vkt/common.hpp>

#include "macros.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKT_FUNC inline void MapVoxelImpl(
            uint8_t* dst,
            float value,
            DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
    {
        value -= mappingLo;
        value /= mappingHi - mappingLo;

        switch (dataFormat)
        {
            case DataFormat::Int16:
            {
                int16_t ival = value * 65535.999f - 32767.f;
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
#else
                dst[0] = static_cast<uint8_t>(ival >> 8);
                dst[1] = static_cast<uint8_t>(ival);
#endif
                break;
            }

            case DataFormat::UInt8:
            {
                uint8_t ival = value * 255.999f;
                dst[0] = ival;
                break;
            }

            case DataFormat::UInt16:
            {
                uint16_t ival = value * 65535.999f;
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
#else
                dst[0] = static_cast<uint8_t>(ival >> 8);
                dst[1] = static_cast<uint8_t>(ival);
#endif
                break;
            }

            case DataFormat::UInt32:
            {
                uint32_t ival = value * 4294967295.999f;
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
                dst[2] = static_cast<uint8_t>(ival >> 16);
                dst[3] = static_cast<uint8_t>(ival >> 24);
#else
                dst[0] = static_cast<uint8_t>(ival >> 24);
                dst[1] = static_cast<uint8_t>(ival >> 16);
                dst[2] = static_cast<uint8_t>(ival >> 8);
                dst[3] = static_cast<uint8_t>(ival);
#endif
                break;
            }

            case DataFormat::Float32:
            {
                uint32_t ival = *(uint32_t*)(&value);
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
                dst[2] = static_cast<uint8_t>(ival >> 16);
                dst[3] = static_cast<uint8_t>(ival >> 24);
#else
                dst[0] = static_cast<uint8_t>(ival >> 24);
                dst[1] = static_cast<uint8_t>(ival >> 16);
                dst[2] = static_cast<uint8_t>(ival >> 8);
                dst[3] = static_cast<uint8_t>(ival);
#endif
                break;
            }
        }
    }

    VKT_FUNC inline void UnmapVoxelImpl(
            float& value,
            uint8_t const* src,
            DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
    {
        switch (dataFormat)
        {
            case DataFormat::Int16:
            {
#ifdef VKT_LITTLE_ENDIAN
                int16_t ival = static_cast<int16_t>(src[0])
                             | static_cast<int16_t>(src[1] << 8);
#else
                int16_t ival = static_cast<int16_t>(src[0] << 8)
                             | static_cast<int16_t>(src[1]);
#endif
                float fval = static_cast<float>(ival);
                value = lerp(mappingLo, mappingHi, (fval + 32767.f) / 65535.999f);
                break;
            }

            case DataFormat::UInt8:
            {
                uint8_t ival = src[0];
                float fval = static_cast<float>(ival);
                value = lerp(mappingLo, mappingHi, fval / 255.999f);
                break;
            }

            case DataFormat::UInt16:
            {
#ifdef VKT_LITTLE_ENDIAN
                uint16_t ival = static_cast<uint16_t>(src[0])
                              | static_cast<uint16_t>(src[1] << 8);
#else
                uint16_t ival = static_cast<uint16_t>(src[0] << 8)
                              | static_cast<uint16_t>(src[1]);
#endif
                float fval = static_cast<float>(ival);
                value = lerp(mappingLo, mappingHi, fval / 65535.999f);
                break;
            }

            case DataFormat::UInt32:
            {
#ifdef VKT_LITTLE_ENDIAN
                uint32_t ival = static_cast<uint32_t>(src[0])
                              | static_cast<uint32_t>(src[1] << 8)
                              | static_cast<uint32_t>(src[2] << 16)
                              | static_cast<uint32_t>(src[3] << 24);
#else
                uint32_t ival = static_cast<uint32_t>(src[0] << 24)
                              | static_cast<uint32_t>(src[1] << 16)
                              | static_cast<uint32_t>(src[2] << 8)
                              | static_cast<uint32_t>(src[3]);
#endif
                float fval = static_cast<float>(ival);
                value = lerp(mappingLo, mappingHi, fval / 4294967295.999f);
                break;
            }

            case DataFormat::Float32:
            {
#ifdef VKT_LITTLE_ENDIAN
                uint32_t ival = static_cast<uint32_t>(src[0])
                              | static_cast<uint32_t>(src[1] << 8)
                              | static_cast<uint32_t>(src[2] << 16)
                              | static_cast<uint32_t>(src[3] << 24);
#else
                uint32_t ival = static_cast<uint32_t>(src[0] << 24)
                              | static_cast<uint32_t>(src[1] << 16)
                              | static_cast<uint32_t>(src[2] << 8)
                              | static_cast<uint32_t>(src[3]);
#endif
                value = *(float*)&ival;
                break;
            }
        }
    }

} // vkt
