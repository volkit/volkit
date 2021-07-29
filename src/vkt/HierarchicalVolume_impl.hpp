// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/HierarchicalVolume.hpp>

struct vktHierarchicalVolume_impl
{
    vktHierarchicalVolume_impl(
            vkt::Brick* bricks,
            std::size_t numBricks,
            vkt::DataFormat dataFormat,
            float mappingLo,
            float mappingHi
            )
        : volume(bricks, numBricks, dataFormat, mappingLo, mappingHi)
    {
    }

    // vktHierarchicalVolume_impl(vkt::HierarchicalVolume& rhs)
    //     : volume(rhs)
    // {
    // }

    vkt::HierarchicalVolume volume;
};
