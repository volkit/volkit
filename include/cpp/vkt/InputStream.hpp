// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    class InputStream
    {
    public:
        InputStream(DataSource& source);

        Error read(StructuredVolume& volume);

        Error readRange(StructuredVolume& dst,
                        int32_t firstX,
                        int32_t firstY,
                        int32_t firstZ,
                        int32_t lastX,
                        int32_t lastY,
                        int32_t lastZ,
                        int32_t dstOffsetX,
                        int32_t dstOffsetY,
                        int32_t dstOffsetZ);

        Error readRange(StructuredVolume& dst,
                        Vec3i first,
                        Vec3i last,
                        Vec3i dstOffset = { 0, 0, 0});

    private:
        DataSource& dataSource_;

    };

} // vkt
