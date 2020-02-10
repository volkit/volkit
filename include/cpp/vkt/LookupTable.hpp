// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    /*!
     * @brief  1D|2D|3D lookup table class
     */
    class LookupTable : public ManagedBuffer<uint8_t>
    {
    public:
        LookupTable(int32_t dimX, int32_t dimY, int32_t dimZ, ColorFormat format);

        void setDims(int32_t dimX, int32_t dimY, int32_t dimZ);
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ);

        void setDims(Vec3i dims);
        Vec3i getDims() const;

        void setColorFormat(ColorFormat pf);
        ColorFormat getColorFormat() const;

        /*!
         * @brief  Provide data buffer: LUT will copy that data to its own memory
         */
        void setData(uint8_t* data);
        uint8_t* getData();

        std::size_t getSizeInBytes() const;

    private:
        Vec3i dims_ = { 0, 0, 0 };
        ColorFormat format_ = ColorFormat::Unspecified;

    };

} // vkt
