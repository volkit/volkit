// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
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
                        int32_t lastZ);

        Error readRange(StructuredVolume& dst,
                        Vec3i first,
                        Vec3i last);

        Error seek(std::size_t pos);

    private:
        DataSource& dataSource_;

    };

} // vkt
