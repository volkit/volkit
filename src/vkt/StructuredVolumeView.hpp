// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>
#include <cstddef>

#include <vkt/common.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include "DataFormatInfo.hpp"
#include "macros.hpp"
#include "VoxelMapping.hpp"

namespace vkt
{
    class StructuredVolumeView
    {
    public:
        VKT_FUNC constexpr static uint8_t GetMaxBytesPerVoxel() { return 8; }

    public:
        StructuredVolumeView() = default;
        VKT_FUNC StructuredVolumeView(StructuredVolume& volume)
            : data_(volume.getData())
            , dims_(volume.getDims())
            , dataFormat_(volume.getDataFormat())
            , dist_(volume.getDist())
            , voxelMapping_(volume.getVoxelMapping())
        {
        }

        VKT_FUNC void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ)
        {
            dimX = dims_.x;
            dimY = dims_.y;
            dimZ = dims_.z;
        }

        VKT_FUNC Vec3i getDims() const
        {
            return dims_;
        }

        VKT_FUNC DataFormat getDataFormat() const
        {
            return dataFormat_;
        }

        VKT_FUNC void getDist(float& distX, float& distY, float& distZ)
        {
            distX = dist_.x;
            distY = dist_.y;
            distZ = dist_.z;
        }

        VKT_FUNC Vec3f getDist() const
        {
            return dist_;
        }

        VKT_FUNC void getVoxelMapping(float& lo, float& hi)
        {
            lo = voxelMapping_.x;
            hi = voxelMapping_.y;
        }

        VKT_FUNC Vec2f getVoxelMapping() const
        {
            return voxelMapping_;
        }

        VKT_FUNC uint8_t const* getData() const
        {
            return data_;
        }

        VKT_FUNC void setValue(int32_t x, int32_t y, int32_t z, float value)
        {
            std::size_t index = linearIndex(x, y, z);

            MapVoxelImpl(
                data_ + index,
                value,
                dataFormat_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

        VKT_FUNC void getValue(int32_t x, int32_t y, int32_t z, float& value)
        {
            std::size_t index = linearIndex(x, y, z);

            UnmapVoxelImpl(
                value,
                data_ + index,
                dataFormat_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

        VKT_FUNC float getValue(int32_t x, int32_t y, int32_t z)
        {
            float value = 0.f;

            getValue(x, y, z, value);

            return value;
        }

        VKT_FUNC void setValue(Vec3i index, float value)
        {
            std::size_t lindex = linearIndex(index);

            MapVoxelImpl(
                data_ + lindex,
                value,
                dataFormat_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

        VKT_FUNC void getValue(Vec3i index, float& value)
        {
            std::size_t lindex = linearIndex(index);

            UnmapVoxelImpl(
                value,
                data_ + lindex,
                dataFormat_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

        VKT_FUNC void setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data)
        {
            std::size_t index = linearIndex(x, y, z);

            for (uint8_t i = 0; i < vkt::getSizeInBytes(dataFormat_); ++i)
                data_[index + i] = data[i];
        }

        VKT_FUNC void getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data)
        {
            std::size_t index = linearIndex(x, y, z);

            for (uint8_t i = 0; i < vkt::getSizeInBytes(dataFormat_); ++i)
                data[i] = data_[index + i];
        }

        VKT_FUNC void setBytes(Vec3i index, uint8_t const* data)
        {
            std::size_t lindex = linearIndex(index);

            for (uint8_t i = 0; i < vkt::getSizeInBytes(dataFormat_); ++i)
                data_[lindex + i] = data[i];
        }

        VKT_FUNC void getBytes(Vec3i index, uint8_t* data)
        {
            std::size_t lindex = linearIndex(index);

            for (uint8_t i = 0; i < vkt::getSizeInBytes(dataFormat_); ++i)
                data[i] = data[lindex + i];
        }

    private:
        uint8_t* data_;
        Vec3i dims_;
        DataFormat dataFormat_;
        Vec3f dist_;
        Vec2f voxelMapping_;


        VKT_FUNC std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const
        {
            size_t index = z * dims_.x * std::size_t(dims_.y)
                         + y * dims_.x
                         + x;
            return index * vkt::getSizeInBytes(dataFormat_);
        }

        VKT_FUNC std::size_t linearIndex(Vec3i index) const
        {
            return linearIndex(index.x, index.y, index.z);
        }

    };

} // vkt
