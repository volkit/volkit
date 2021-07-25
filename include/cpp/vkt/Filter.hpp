// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

/*! @file  Filter.hpp
 * @brief  ApplyFilter algorithm, applies a convolution filter to volumes
 */
namespace vkt
{
    //! Filter class, data is managed
    class Filter : public ManagedBuffer<std::size_t>
    {
    public:
        Filter(float* data, Vec3i dims);

        float* getData();

        Vec3i getDims() const;

    private:
        Vec3i dims_;
    };

    //! Address mode to use at boundaries
    enum class AddressMode
    {
        Wrap,
        Mirror,
        Clamp,
        Border,
    };

    //! Apply filter to the source volume, store the result in dest
    VKTAPI Error ApplyFilter(StructuredVolume& dest,
                             StructuredVolume& source,
                             Filter& filter,
                             AddressMode am);


} // vkt
