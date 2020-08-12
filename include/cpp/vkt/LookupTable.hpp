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
    /*! @class LookupTable
     * @brief  1D|2D|3D lookup table class
     *
     * This class represents a lookup table in memory. The lookup table is
     * managed, i.e., where the actual data is stored (e.g., on the CPU or on
     * the GPU) is determined by the internal memory management layer. This can
     * be influenced by setting the active thread's execution policy via @ref
     * vkt::SetThreadExecutionPolicy(). When the execution policy was changed
     * so that the lookup table should be present on another device, the
     * algorithm accessing the data next will initiate for the lookup table to
     * be migrated before the respective operation is performed. This happens
     * in a deferred fashion.
     */
    class LookupTable : public ManagedBuffer<uint8_t>
    {
    public:
        LookupTable();
        LookupTable(int32_t dimX, int32_t dimY, int32_t dimZ, ColorFormat format);

        void setDims(int32_t dimX, int32_t dimY, int32_t dimZ);
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ);

        void setDims(Vec3i dims);
        Vec3i getDims() const;

        void setColorFormat(ColorFormat cf);
        ColorFormat getColorFormat() const;

        /*!
         * @brief  Provide data buffer: LUT will copy that data to its own memory
         */
        void setData(uint8_t* data);

        /*!
         * @brief  Provide data buffer, data points will be resampled
         */
        void setData(uint8_t* data,
                     int32_t dimX,
                     int32_t dimY,
                     int32_t dimZ,
                     ColorFormat sourceFormat);

        uint8_t* getData();

        std::size_t getSizeInBytes() const;

    private:
        Vec3i dims_ = { 0, 0, 0 };
        ColorFormat format_ = ColorFormat::Unspecified;

    };

} // vkt
