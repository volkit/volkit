// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "linalg.hpp"

namespace vkt
{
    class StructuredVolume : public ManagedBuffer<uint8_t>
    {
    public:
        constexpr static uint16_t GetMaxBytesPerVoxel() { return 4; }
    
    public:
        StructuredVolume();
        StructuredVolume(
                int32_t dimX,
                int32_t dimY,
                int32_t dimZ,
                uint16_t bytesPerVoxel,
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

        void setBytesPerVoxel(uint16_t bpv);
        uint16_t getBytesPerVoxel() const;

        void setDist(float distX, float distY, float distZ);
        void getDist(float& distX, float& distY, float& distZ);

        void setDist(Vec3f dist);
        Vec3f getDist() const;

        void setVoxelMapping(float lo, float hi);
        void getVoxelMapping(float& lo, float& hi);

        void setVoxelMapping(Vec2f mapping);
        Vec2f getVoxelMapping() const;

        uint8_t* getData();

        void setValue(int32_t x, int32_t y, int32_t z, float value);
        void getValue(int32_t x, int32_t y, int32_t z, float& value);

        void setValue(Vec3i index, float value);
        void getValue(Vec3i index, float& value);

        void setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data);
        void getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data);

        void setBytes(Vec3i index, uint8_t const* data);
        void getBytes(Vec3i index, uint8_t* data);

        std::size_t getSizeInBytes() const;

    private:
        Vec3i dims_;
        uint16_t bytesPerVoxel_;
        Vec3f dist_;
        Vec2f voxelMapping_;

        std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const;
        std::size_t linearIndex(Vec3i index) const;
    };
} // vkt
