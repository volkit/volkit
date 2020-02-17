// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>
#include <cstddef>

#include <vkt/common.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include "macros.hpp"
#include "VoxelMapping.hpp"

namespace vkt
{
    class StructuredVolumeView
    {
    public:
        VKT_FUNC constexpr static uint16_t GetMaxBytesPerVoxel() { return 8; }

    public:
        StructuredVolumeView() = default;
        VKT_FUNC StructuredVolumeView(StructuredVolume& volume)
            : data_(volume.getData())
            , dims_(volume.getDims())
            , bytesPerVoxel_(volume.getBytesPerVoxel())
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

        VKT_FUNC uint16_t getBytesPerVoxel() const
        {
            return bytesPerVoxel_;
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
                bytesPerVoxel_,
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
                bytesPerVoxel_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

        VKT_FUNC void setValue(Vec3i index, float value)
        {
            std::size_t lindex = linearIndex(index);

            MapVoxelImpl(
                data_ + lindex,
                value,
                bytesPerVoxel_,
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
                bytesPerVoxel_,
                voxelMapping_.x,
                voxelMapping_.y
                );
        }

    private:
        uint8_t* data_;
        Vec3i dims_;
        uint16_t bytesPerVoxel_;
        Vec3f dist_;
        Vec2f voxelMapping_;


        VKT_FUNC std::size_t linearIndex(int32_t x, int32_t y, int32_t z) const
        {
            size_t index = z * dims_.x * std::size_t(dims_.y)
                         + y * dims_.x
                         + x;
            return index * bytesPerVoxel_;
        }

        VKT_FUNC std::size_t linearIndex(Vec3i index) const
        {
            return linearIndex(index.x, index.y, index.z);
        }

    };

} // vkt
