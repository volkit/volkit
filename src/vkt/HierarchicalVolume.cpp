// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>

#include <vkt/HierarchicalVolume.hpp>

#include "DataFormatInfo.hpp"
#include "linalg.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    HierarchicalVolume::HierarchicalVolume(
            Brick const* bricks,
            std::size_t numBricks,
            DataFormat dataFormat
            )
        : bricks_(numBricks)
        , dataFormat_(dataFormat)
    {
        std::memcpy(bricks_.data(), bricks, numBricks * sizeof(Brick));

        std::size_t newSize = 0;
        for (std::size_t i = 0; i < bricks_.numElements(); ++i)
        {
            Vec3i dims = bricks_[i].dims;
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

        for (std::size_t i = 0; i < bricks_.numElements(); ++i)
        {
            Vec3i lo{0,0,0};
            Vec3i hi = bricks_[i].dims;
            lo *= (int)(1<<bricks_[i].level);
            hi *= (int)(1<<bricks_[i].level);
            lo += bricks_[i].lower;
            hi += bricks_[i].lower;

            lower = min(lower, lo);
            upper = max(upper, hi);
        }

        dimX = upper.x - lower.x;
        dimY = upper.y - lower.y;
        dimZ = upper.z - lower.z;
    }

    std::size_t HierarchicalVolume::getNumBricks()
    {
        return bricks_.numElements();
    }

    Brick* HierarchicalVolume::getBricks()
    {
        return bricks_.data();
    }

    DataFormat HierarchicalVolume::getDataFormat() const
    {
        return dataFormat_;
    }

    uint8_t* HierarchicalVolume::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }
} // vkt
