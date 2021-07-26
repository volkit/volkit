// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstring> // memcpy

#include <vkt/StructuredVolume.hpp>

#include <vkt/StructuredVolume.h>
#include <vkt/System.h>

#include "DataFormatInfo.hpp"
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
        , dataFormat_(DataFormat::UInt8)
        , dist_{1.f, 1.f, 1.f}
        , voxelMapping_{0.f, 1.f}
        , haloSize_{.5f, .5f, .5f}
    {
    }

    StructuredVolume::StructuredVolume(
            int32_t dimX,
            int32_t dimY,
            int32_t dimZ,
            DataFormat dataFormat,
            float distX,
            float distY,
            float distZ,
            float mappingLo,
            float mappingHi
            )
        : ManagedBuffer(dimX * size_t(dimY) * dimZ * vkt::getSizeInBytes(dataFormat))
        , dims_{dimX, dimY, dimZ}
        , dataFormat_(dataFormat)
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

    void StructuredVolume::setDataFormat(DataFormat dataFormat)
    {
        dataFormat_ = dataFormat;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    DataFormat StructuredVolume::getDataFormat() const
    {
        return dataFormat_;
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

    Box3f StructuredVolume::getDomainBounds() const
    {
        Box3f domainBounds = getObjectBounds();

        domainBounds.min -= haloSize_;
        domainBounds.max += haloSize_;

        return domainBounds;
    }

    Box3f StructuredVolume::getObjectBounds() const
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

    void StructuredVolume::setValue(int32_t x, int32_t y, int32_t z, float value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        MapVoxelImpl(
            ManagedBuffer::data_ + index,
            value,
            dataFormat_,
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
            dataFormat_,
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
            dataFormat_,
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
            dataFormat_,
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
            dataFormat_,
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
            dataFormat_,
            voxelMapping_.x,
            voxelMapping_.y
            );

        return value;
    }

    void StructuredVolume::setBytes(int32_t x, int32_t y, int32_t z, uint8_t const* data)
    {
        migrate();

        std::size_t index = linearIndex(x, y, z);

        for (uint8_t i = 0; i < getBytesPerVoxel(); ++i)
            ManagedBuffer::data_[index + i] = data[i];
    }

    void StructuredVolume::getBytes(int32_t x, int32_t y, int32_t z, uint8_t* data)
    {
        migrate();

        std::size_t index = linearIndex(x, y, z);

        for (uint8_t i = 0; i < getBytesPerVoxel(); ++i)
            data[i] = ManagedBuffer::data_[index + i];
    }

    void StructuredVolume::setBytes(Vec3i index, uint8_t const* data)
    {
        migrate();

        std::size_t lindex = linearIndex(index);

        for (uint8_t i = 0; i < getBytesPerVoxel(); ++i)
            ManagedBuffer::data_[lindex + i] = data[i];
    }

    void StructuredVolume::getBytes(Vec3i index, uint8_t* data)
    {
        migrate();

        std::size_t lindex = linearIndex(index);

        for (uint8_t i = 0; i < getBytesPerVoxel(); ++i)
            data[i] = ManagedBuffer::data_[lindex + i];
    }

    uint8_t StructuredVolume::getBytesPerVoxel() const
    {
        return vkt::getSizeInBytes(dataFormat_);
    }

    std::size_t StructuredVolume::getSizeInBytes() const
    {
        return dims_.x * std::size_t(dims_.y) * dims_.z * getBytesPerVoxel();
    }


    //--- private -----------------------------------------

    std::size_t StructuredVolume::linearIndex(int32_t x, int32_t y, int32_t z) const
    {
        size_t index = z * dims_.x * std::size_t(dims_.y)
                     + y * dims_.x
                     + x;
        return index * getBytesPerVoxel();
    }

    std::size_t StructuredVolume::linearIndex(Vec3i index) const
    {
        return linearIndex(index.x, index.y, index.z);
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

uint8_t vktStructuredVolumeGetMaxBytesPerVoxel()
{
    return vkt::StructuredVolume::GetMaxBytesPerVoxel();
}

void vktStructuredVolumeCreate(
        vktStructuredVolume* volume,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ,
        vktDataFormat dataFormat,
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
            (vkt::DataFormat)dataFormat,
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
