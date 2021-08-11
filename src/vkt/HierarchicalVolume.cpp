// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstdlib>

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/Memory.hpp>

#include <vkt/HierarchicalVolume.h>

#include "DataFormatInfo.hpp"
#include "linalg.hpp"
#include "HierarchicalVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    HierarchicalVolume::HierarchicalVolume(
            Brick const* bricks,
            std::size_t numBricks,
            DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
        : bricks_(numBricks)
        , dataFormat_(dataFormat)
        , voxelMapping_{mappingLo, mappingHi}
    {
        ExecutionPolicy ep = GetThreadExecutionPolicy();
        CopyKind ck = CopyKind::HostToHost;

        if (ep.device == ExecutionPolicy::Device::GPU)
            ck = CopyKind::HostToDevice;

        Memcpy(bricks_.data(), bricks, numBricks * sizeof(Brick), ck);

        std::size_t newSize = 0;
        for (std::size_t i = 0; i < numBricks; ++i)
        {
            Vec3i dims = bricks[i].dims;
            newSize += dims.x * dims.y * dims.z *  vkt::getSizeInBytes(dataFormat_);
        }
        resize(newSize);
    }

    Vec3i HierarchicalVolume::getDims() const
    {
        int32_t dimX;
        int32_t dimY;
        int32_t dimZ;
        getDims(dimX, dimY, dimZ);
        return { dimX, dimY, dimZ };
    }

    void HierarchicalVolume::getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ) const
    {
        Vec3i lower{INT_MAX,INT_MAX,INT_MAX};
        Vec3i upper{INT_MIN,INT_MIN,INT_MIN};

        Brick* bricks = nullptr;

        ExecutionPolicy ep = GetThreadExecutionPolicy();
        if (ep.device == ExecutionPolicy::Device::GPU)
        {
            bricks = new Brick[bricks_.numElements()];
            Memcpy(bricks, bricks_.data(), bricks_.numElements() * sizeof(Brick),
                   CopyKind::DeviceToHost);
        }
        else
            bricks = (Brick*)bricks_.data();

        for (std::size_t i = 0; i < bricks_.numElements(); ++i)
        {
            Vec3i lo{0,0,0};
            Vec3i hi = bricks[i].dims;
            lo *= (int)(1<<bricks[i].level);
            hi *= (int)(1<<bricks[i].level);
            lo += bricks[i].lower;
            hi += bricks[i].lower;

            lower = min(lower, lo);
            upper = max(upper, hi);
        }

        dimX = std::min(maxDims_.x, upper.x - lower.x);
        dimY = std::min(maxDims_.y, upper.y - lower.y);
        dimZ = std::min(maxDims_.z, upper.z - lower.z);

        if (ep.device == ExecutionPolicy::Device::GPU)
            delete[] bricks;
    }

    std::size_t HierarchicalVolume::getNumBricks()
    {
        return bricks_.numElements();
    }

    void HierarchicalVolume::setBricks(Brick const* bricks, std::size_t numBricks)
    {
        bricks_.resize(numBricks);

        ExecutionPolicy ep = GetThreadExecutionPolicy();
        CopyKind ck = CopyKind::HostToHost;

        if (ep.device == ExecutionPolicy::Device::GPU)
            ck = CopyKind::HostToDevice;

        Memcpy(bricks_.data(), bricks, numBricks * sizeof(Brick), ck);

        std::size_t newSize = 0;
        for (std::size_t i = 0; i < numBricks; ++i)
        {
            Vec3i dims = bricks[i].dims;
            newSize += dims.x * dims.y * dims.z *  vkt::getSizeInBytes(dataFormat_);
        }
        resize(newSize);
    }

    void HierarchicalVolume::setMaxDims(int dimX, int dimY, int dimZ)
    {
        maxDims_ = { dimX, dimY, dimZ };
    }

    void HierarchicalVolume::setMaxDims(Vec3i maxDims)
    {
        maxDims_ = maxDims;
    }

    Brick* HierarchicalVolume::getBricks()
    {
        return bricks_.data();
    }

    DataFormat HierarchicalVolume::getDataFormat() const
    {
        return dataFormat_;
    }

    void HierarchicalVolume::setVoxelMapping(float lo, float hi)
    {
        voxelMapping_ = {lo, hi };
    }

    void HierarchicalVolume::getVoxelMapping(float& lo, float& hi)
    {
        lo = voxelMapping_.x;
        hi = voxelMapping_.y;
    }

    void HierarchicalVolume::setVoxelMapping(Vec2f mapping)
    {
        voxelMapping_ = mapping;
    }

    Vec2f HierarchicalVolume::getVoxelMapping() const
    {
        return voxelMapping_;
    }

    uint8_t* HierarchicalVolume::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }
} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

uint8_t vktHierarchicalVolumeGetMaxBytesPerVoxel()
{
    return vkt::HierarchicalVolume::GetMaxBytesPerVoxel();
}

void vktHierarchicalVolumeCreate(
        vktHierarchicalVolume* volume,
        vktBrick_t* bricks,
        size_t numBricks,
        vktDataFormat dataFormat,
        float mappingLo,
        float mappingHi
        )
{
    assert(volume != nullptr);

    *volume = new vktHierarchicalVolume_impl(
            (vkt::Brick*)bricks,
            numBricks,
            (vkt::DataFormat)dataFormat,
            mappingLo,
            mappingHi
            );
}

void vktHierarchicalVolumeDestroy(vktHierarchicalVolume volume)
{
    delete volume;
}
