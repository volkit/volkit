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
                int32_t dimx,
                int32_t dimy,
                int32_t dimz,
                uint16_t bytesPerVoxel,
                float mappingLo = 0.f,
                float mappingHi = 1.f
                );
        StructuredVolume(StructuredVolume& rhs);
        StructuredVolume(StructuredVolume&& rhs) = default;
       ~StructuredVolume();

        StructuredVolume& operator=(StructuredVolume& rhs);
        StructuredVolume& operator=(StructuredVolume&& rhs) = default;

        void setDims(int32_t dimx, int32_t dimy, int32_t dimz);
        void getDims(int32_t& dimx, int32_t& dimy, int32_t& dimz);

        void setDims(vec3i dims);
        vec3i getDims() const;

        void setBytesPerVoxel(uint16_t bpv);
        uint16_t getBytesPerVoxel() const;

        void setVoxelMapping(float lo, float hi);
        void getVoxelMapping(float& lo, float& hi);

        void setVoxelMapping(vec2f mapping);
        vec2f getVoxelMapping() const;

        uint8_t* getData();

        void setValue(int32_t x, int32_t y, int32_t z, float value);
        void getValue(int32_t x, int32_t y, int32_t z, float& value);

        void setVoxel(int32_t x, int32_t y, int32_t z, uint8_t const* data);
        void getVoxel(int32_t x, int32_t y, int32_t z, uint8_t* data);

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
        vec2f voxelMapping_;

        std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const;
    };
} // vkt
