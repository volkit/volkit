// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstring> // memcpy

#include <vkt/StructuredVolume.hpp>

#include <vkt/StructuredVolume.h>
#include <vkt/System.h>

#include "linalg.hpp"
#include "StructuredVolume_impl.hpp"
#include "VoxelMapping.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    StructuredVolume::StructuredVolume()
        : ManagedBuffer(0)
        , dims_{0, 0 , 0}
        , bytesPerVoxel_(1)
        , dist_{1.f, 1.f, 1.f}
        , voxelMapping_{0.f, 1.f}
    {
    }

    StructuredVolume::StructuredVolume(
            int32_t dimX,
            int32_t dimY,
            int32_t dimZ,
            uint16_t bytesPerVoxel,
            float distX,
            float distY,
            float distZ,
            float mappingLo,
            float mappingHi
            )
        : ManagedBuffer(dimX * size_t(dimY) * dimZ * bytesPerVoxel)
        , dims_{dimX, dimY, dimZ}
        , bytesPerVoxel_(bytesPerVoxel)
        , dist_{distX, distY, distZ}
        , voxelMapping_{mappingLo, mappingHi}
    {
    }

    void StructuredVolume::setDims(int32_t dimX, int32_t dimY, int32_t dimZ)
    {
        setDims({ dimX, dimY, dimZ });
    }

    void StructuredVolume::getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ)
    {
        dimX = dims_.x;
        dimY = dims_.y;
        dimZ = dims_.z;
    }

    void StructuredVolume::setDims(Vec3i dims)
    {
        dims_ = dims;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    Vec3i StructuredVolume::getDims() const
    {
        return dims_;
    }

    void StructuredVolume::setBytesPerVoxel(uint16_t bpv)
    {
        bytesPerVoxel_ = bpv;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    uint16_t StructuredVolume::getBytesPerVoxel() const
    {
        return bytesPerVoxel_;
    }

    void StructuredVolume::setDist(float distX, float distY, float distZ)
    {
        dist_ = { distX, distY, distZ };
    }

    void StructuredVolume::getDist(float& distX, float& distY, float& distZ)
    {
        distX = dist_.x;
        distY = dist_.y;
        distZ = dist_.z;
    }

    void StructuredVolume::setDist(Vec3f dist)
    {
        dist_ = dist;
    }

    Vec3f StructuredVolume::getDist() const
    {
        return dist_;
    }

    void StructuredVolume::setVoxelMapping(float lo, float hi)
    {
        voxelMapping_ = {lo, hi };
    }

    void StructuredVolume::getVoxelMapping(float& lo, float& hi)
    {
        lo = voxelMapping_.x;
        hi = voxelMapping_.y;
    }

    void StructuredVolume::setVoxelMapping(Vec2f mapping)
    {
        voxelMapping_ = mapping;
    }

    Vec2f StructuredVolume::getVoxelMapping() const
    {
        return voxelMapping_;
    }

    Box3f StructuredVolume::getWorldBounds() const
    {
        return {
            { 0.f, 0.f, 0.f },
            { dims_.x * dist_.x, dims_.y * dist_.y, dims_.z * dist_.z }
            };
    }

    uint8_t* StructuredVolume::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }

    float StructuredVolume::sampleLinear(int32_t x, int32_t y, int32_t z)
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

    void StructuredVolume::setValue(int32_t x, int32_t y, int32_t z, float value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        MapVoxelImpl(
            ManagedBuffer::data_ + index,
            value,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
    }

    void StructuredVolume::getValue(int32_t x, int32_t y, int32_t z, float& value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        UnmapVoxelImpl(
            value,
            ManagedBuffer::data_ + index,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
    }

    float StructuredVolume::getValue(int32_t x, int32_t y, int32_t z)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        float value = 0.f;

        UnmapVoxelImpl(
            value,
            ManagedBuffer::data_ + index,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );

        return value;
    }

    void StructuredVolume::setValue(Vec3i index, float value)
    {
        migrate();

        size_t lindex = linearIndex(index);

        MapVoxelImpl(
            ManagedBuffer::data_ + lindex,
            value,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
    }

    void StructuredVolume::getValue(Vec3i index, float& value)
    {
        migrate();

        size_t lindex = linearIndex(index);

        UnmapVoxelImpl(
            value,
            ManagedBuffer::data_ + lindex,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
    }

    float StructuredVolume::getValue(Vec3i index)
    {
        migrate();

        size_t lindex = linearIndex(index);

        float value = 0.f;

        UnmapVoxelImpl(
            value,
            ManagedBuffer::data_ + lindex,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );

        return value;
    }

    void StructuredVolume::setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            ManagedBuffer::data_[index + i] = data[i];
    }

    void StructuredVolume::getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            data[i] = ManagedBuffer::data_[index + i];
    }

    void StructuredVolume::setBytes(Vec3i index, uint8_t const* data)
    {
        migrate();

        size_t lindex = linearIndex(index);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            ManagedBuffer::data_[lindex + i] = data[i];
    }

    void StructuredVolume::getBytes(Vec3i index, uint8_t* data)
    {
        migrate();

        size_t lindex = linearIndex(index);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            data[i] = ManagedBuffer::data_[lindex + i];
    }

    std::size_t StructuredVolume::getSizeInBytes() const
    {
        return dims_.x * std::size_t(dims_.y) * dims_.z * bytesPerVoxel_;
    }


    //--- private -----------------------------------------

    std::size_t StructuredVolume::linearIndex(int32_t x, int32_t y, int32_t z) const
    {
        size_t index = z * dims_.x * std::size_t(dims_.y)
                     + y * dims_.x
                     + x;
        return index * bytesPerVoxel_;
    }

    std::size_t StructuredVolume::linearIndex(Vec3i index) const
    {
        return linearIndex(index.x, index.y, index.z);
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

uint16_t vktStructuredVolumeGetMaxBytesPerVoxel()
{
    return vkt::StructuredVolume::GetMaxBytesPerVoxel();
}

void vktStructuredVolumeCreate(
        vktStructuredVolume* volume,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ,
        uint16_t bytesPerVoxel,
        float distX,
        float distY,
        float distZ,
        float mappingLo,
        float mappingHi
        )
{
    assert(volume != nullptr);

    *volume = new vktStructuredVolume_impl(
            dimX,
            dimY,
            dimZ,
            bytesPerVoxel,
            distX,
            distY,
            distZ,
            mappingLo,
            mappingHi
            );
}

void vktStructuredVolumeCreateCopy(vktStructuredVolume* volume, vktStructuredVolume rhs)
{
    assert(volume != nullptr);

    *volume = new vktStructuredVolume_impl(rhs->volume);
}

void vktStructuredVolumeDestroy(vktStructuredVolume volume)
{
    delete volume;
}
