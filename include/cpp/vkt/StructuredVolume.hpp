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
    /*! @class StructuredVolume
     * @brief  Managed structured volume class
     *
     * This class represents a structured volume in memory. The volume is
     * managed, i.e., where the actual data is stored (e.g., on the CPU or on
     * the GPU) is determined by the internal memory management layer. This can
     * be influenced by setting the active thread's execution policy via @ref
     * vkt::SetThreadExecutionPolicy(). When the execution policy was changed
     * so that the volume data should be present on another device, the
     * algorithm accessing the data next will initiate for the volume to be
     * migrated before the respective operation is performed. This happens in a
     * deferred fashion.
     *
     * Structured volumes have 3D dimensions, store the data format, the
     * distance between voxels in x,y, and z direction, as well as a linear
     * mapping from the minimum and maximum data value to the floating point
     * range `[lo..hi]`.
     */
    class StructuredVolume : public ManagedBuffer<uint8_t>
    {
    public:
        constexpr static uint16_t GetMaxBytesPerVoxel() { return 8; }
    
    public:
        StructuredVolume();
        StructuredVolume(
                int32_t dimX,
                int32_t dimY,
                int32_t dimZ,
                DataFormat dataFormat,
                float distX = 1.f,
                float distY = 1.f,
                float distZ = 1.f,
                float mappingLo = 0.f,
                float mappingHi = 1.f
                );
        StructuredVolume(StructuredVolume& rhs) = default;
        StructuredVolume(StructuredVolume&& rhs) = default;

        StructuredVolume& operator=(StructuredVolume& rhs) = default;
        StructuredVolume& operator=(StructuredVolume&& rhs) = default;

        void setDims(int32_t dimX, int32_t dimY, int32_t dimZ);
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ);

        void setDims(Vec3i dims);
        Vec3i getDims() const;

        void setDataFormat(DataFormat dataFormat);
        DataFormat getDataFormat() const;

        void setDist(float distX, float distY, float distZ);
        void getDist(float& distX, float& distY, float& distZ);

        void setDist(Vec3f dist);
        Vec3f getDist() const;

        void setVoxelMapping(float lo, float hi);
        void getVoxelMapping(float& lo, float& hi);

        void setVoxelMapping(Vec2f mapping);
        Vec2f getVoxelMapping() const;

        Box3f getWorldBounds() const;

        uint8_t* getData();

        float sampleLinear(int32_t x, int32_t y, int32_t z);

        void setValue(int32_t x, int32_t y, int32_t z, float value);
        void getValue(int32_t x, int32_t y, int32_t z, float& value);
        float getValue(int32_t x, int32_t y, int32_t z);

        void setValue(Vec3i index, float value);
        void getValue(Vec3i index, float& value);
        float getValue(Vec3i index);

        void setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data);
        void getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data);

        void setBytes(Vec3i index, uint8_t const* data);
        void getBytes(Vec3i index, uint8_t* data);

        uint8_t getBytesPerVoxel() const;
        std::size_t getSizeInBytes() const;

    private:
        Vec3i dims_;
        DataFormat dataFormat_;
        Vec3f dist_;
        Vec2f voxelMapping_;

        std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const;
        std::size_t linearIndex(Vec3i index) const;
    };
} // vkt
