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
    class OutputStream
    {
    public:
        OutputStream(DataSource& source);

        Error write(StructuredVolume& volume);

        Error writeRange(StructuredVolume& dst,
                         int32_t firstX,
                         int32_t firstY,
                         int32_t firstZ,
                         int32_t lastX,
                         int32_t lastY,
                         int32_t lastZ);

        Error writeRange(StructuredVolume& dst,
                         Vec3i first,
                         Vec3i last);

        Error seek(std::size_t pos);

        Error flush();

    private:
        DataSource& dataSource_;

    };

} // vkt
