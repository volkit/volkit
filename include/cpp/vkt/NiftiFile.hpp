// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    class NiftiFile : public DataSource
    {
    public:
        NiftiFile(char const* fileName);
       ~NiftiFile();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual bool seek(std::size_t pos);
        virtual bool good() const;

        Vec3i getDims();

        uint16_t getBytesPerVoxel();

        Vec2f getVoxelMapping();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

    };

} // vkt
