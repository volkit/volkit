#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "linalg.hpp"

namespace vkt
{
    class StructuredVolume : public ManagedBuffer
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
        StructuredVolume(StructuredVolume& rhs);
        StructuredVolume(StructuredVolume&& rhs) = default;
       ~StructuredVolume();

        StructuredVolume& operator=(StructuredVolume& rhs);
        StructuredVolume& operator=(StructuredVolume&& rhs) = default;

        void setDims(int32_t dimX, int32_t dimY, int32_t dimZ);
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ);

        void setDims(vec3i dims);
        vec3i getDims() const;

        void setBytesPerVoxel(uint16_t bpv);
        uint16_t getBytesPerVoxel() const;

        void setDist(float distX, float distY, float distZ);
        void getDist(float& distX, float& distY, float& distZ);

        void setDist(vec3f dist);
        vec3f getDist() const;

        void setVoxelMapping(float lo, float hi);
        void getVoxelMapping(float& lo, float& hi);

        void setVoxelMapping(vec2f mapping);
        vec2f getVoxelMapping() const;

        uint8_t* getData();

        void setValue(int32_t x, int32_t y, int32_t z, float value);
        void getValue(int32_t x, int32_t y, int32_t z, float& value);

        void setValue(vec3i index, float value);
        void getValue(vec3i index, float& value);

        void setVoxel(int32_t x, int32_t y, int32_t z, uint8_t const* data);
        void getVoxel(int32_t x, int32_t y, int32_t z, uint8_t* data);

        void setVoxel(vec3i index, uint8_t const* data);
        void getVoxel(vec3i index, uint8_t* data);

        /*!
         * @brief  provide in-memory representation of mapped voxel
         */
        void mapVoxel(uint8_t* dst, float src) const;

        /*!
         * @brief  convert from mapped in-memory representation to voxel value
         */
        void unmapVoxel(float& dst, uint8_t const* src) const;

        std::size_t getSizeInBytes() const;

    private:
        vec3i dims_;
        uint16_t bytesPerVoxel_;
        vec3f dist_;
        vec2f voxelMapping_;

        std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const;
        std::size_t linearIndex(vec3i index) const;
    };
} // vkt
