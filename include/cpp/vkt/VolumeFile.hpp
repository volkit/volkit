// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    struct VolumeFileHeader
    {
        bool isStructured = false;
        Vec3i dims = { 0, 0, 0 };
        uint16_t bytesPerVoxel = 0;
    };

    class VolumeFile : public DataSource
    {
    public:

        VolumeFile(char const* fileName);
       ~VolumeFile();

        std::size_t read(char* buf, std::size_t len);
        bool good() const;

        VolumeFileHeader getHeader() const;

    private:
        VolumeFileHeader header_;

        DataSource* dataSource_ = nullptr;

    };

} // vkt
