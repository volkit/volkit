// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/Array1D.hpp>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    struct Brick
    {
        Vec3i lower;
        Vec3i dims;
        std::size_t offsetInBytes;
        unsigned level;
    };

    /*! @class HierarchicalVolume
     * @brief  Managed hierarchical volume class
     */
    class HierarchicalVolume : public ManagedBuffer<uint8_t>
    {
    public:
        HierarchicalVolume() = default;

        HierarchicalVolume(
                Brick const* bricks,
                std::size_t numBricks,
                DataFormat dataFormat);

        //! Compute logical grid size
        Vec3i getDims() const;

        //! Compute logical grid size
        void getDims(int32_t& dimX, int32_t& dimY, int32_t& dimZ) const;

        std::size_t getNumBricks();

        Brick* getBricks();

        DataFormat getDataFormat() const;

        //! Get a raw pointer to the internal data
        uint8_t* getData();

    private:
        Array1D<Brick> bricks_;
        DataFormat dataFormat_;
    };
} // vkt
