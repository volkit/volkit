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

        VKT_FUNC float sampleLinear(int32_t x, int32_t y, int32_t z)
        {
            float xf1 = x - 0.f;
            float yf1 = y - 0.f;
            float zf1 = z - 0.f;

            float xf2 = x + 1.f;
            float yf2 = y + 1.f;
            float zf2 = z + 1.f;

            Vec3i lo{ (int)xf1, (int)yf1, (int)zf1 };
            Vec3i hi{ (int)xf2, (int)yf2, (int)zf2 };

            lo.x = clamp(lo.x, 0, dims_.x - 1);
            lo.y = clamp(lo.y, 0, dims_.y - 1);
            lo.x = clamp(lo.x, 0, dims_.x - 1);

            hi.y = clamp(hi.y, 0, dims_.y - 1);
            hi.z = clamp(hi.z, 0, dims_.z - 1);
            hi.z = clamp(hi.z, 0, dims_.z - 1);

            Vec3f frac{ xf1 - lo.x, yf1 - lo.y, zf1 - lo.z };

            float v[8] = {
                getValue(lo.x, lo.y, lo.z),
                getValue(hi.x, lo.y, lo.z),
                getValue(lo.x, hi.y, lo.z),
                getValue(hi.x, hi.y, lo.z),
                getValue(lo.x, lo.y, hi.z),
                getValue(hi.x, lo.y, hi.z),
                getValue(lo.x, hi.y, hi.z),
                getValue(hi.x, hi.y, hi.z)
                };

            return lerp(
                lerp(lerp(v[0], v[1], frac.x), lerp(v[2], v[3], frac.x), frac.y),
                lerp(lerp(v[4], v[5], frac.x), lerp(v[6], v[7], frac.x), frac.y),
                frac.z
                );
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

        VKT_FUNC void getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data) const
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

        VKT_FUNC void getBytes(Vec3i index, uint8_t* data) const
        {
            std::size_t lindex = linearIndex(index);

            for (uint8_t i = 0; i < vkt::getSizeInBytes(dataFormat_); ++i)
                data[i] = data_[lindex + i];
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
