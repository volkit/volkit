// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstring> // memcpy

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/Voxel.hpp>

#include <vkt/StructuredVolume.h>
#include <vkt/System.h>

#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    StructuredVolume::StructuredVolume()
        : ManagedBuffer(0)
        , dims_(0, 0 , 0)
        , bytesPerVoxel_(1)
        , dist_(1.f, 1.f, 1.f)
        , voxelMapping_(0.f, 1.f)
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
        , dims_(dimX, dimY, dimZ)
        , bytesPerVoxel_(bytesPerVoxel)
        , dist_(distX, distY, distZ)
        , voxelMapping_(mappingLo, mappingHi)
    {
    }

    StructuredVolume::~StructuredVolume()
    {
    }

    void StructuredVolume::setDims(int32_t dimX, int32_t dimY, int32_t dimZ)
    {
        setDims(Vec3i(dimX, dimY, dimZ));
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
        dist_ = Vec3f(distX, distY, distZ);
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
        voxelMapping_ = Vec2f(lo, hi);
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

    uint8_t* StructuredVolume::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }

    void StructuredVolume::setValue(int32_t x, int32_t y, int32_t z, float value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        MapVoxel(
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

        UnmapVoxel(
            value,
            ManagedBuffer::data_ + index,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
    }

    void StructuredVolume::setValue(Vec3i index, float value)
    {
        migrate();

        size_t lindex = linearIndex(index);

        MapVoxel(
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

        UnmapVoxel(
            value,
            ManagedBuffer::data_ + lindex,
            bytesPerVoxel_,
            voxelMapping_.x,
            voxelMapping_.y
            );
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
